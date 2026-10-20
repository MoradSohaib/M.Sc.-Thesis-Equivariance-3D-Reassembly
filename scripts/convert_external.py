#!/usr/bin/env python3
"""
Convert Fractura (GARF's HDF5) and Fantastic Breaks into Breaking Bad's
layout, so the model is evaluated on them exactly as on Breaking Bad.

    python -m scripts.convert_external Fractura/bone_real.hdf5 path/to/fantastic_breaks \\
        --out external --reference data
    python -m scripts.convert_external Fractura/<synthetic>.hdf5 --split val \\
        --out external --reference data

Each source becomes one subset, named after the file or folder (``-`` becomes
``_``); then, per subset::

    python -m scripts.train --root_dir external --evaluate --checkpoint path/to/best.pt \\
        --data_subsets bone_real --split all --modes_per_scene 0 --max_fragments 20 \\
        --checkpoint_dir eval/bone_real --predictions results/bone_real

Every assembly is read with its pieces in the assembled pose, then
(``reassembly.data.external.prepare``):

1. cleaned to one closed surface per piece -- vertices within 1e-6 of the
   piece's size merged, the doubled interior walls of simulated pieces
   cancelled, a copy of another piece stored inside a piece removed (Fractura's
   vert8), surfaces wound inward turned outward, and, where the pieces do not
   share their breaks (scans), separate debris dropped (Blender's default
   cube, in every real bone piece checked);
2. scaled so the object's largest side is 1, as Breaking Bad's objects are
   (``--scale piece``: GARF's loader instead, scaled down until the largest
   piece is at most 1);
3. thinned, by quadric edge collapse with one face budget per object split
   over its pieces by area, to Breaking Bad's density: ``--reference DATA``
   measures it on your Breaking Bad scenes (the median edge over the object's
   largest side); ``--edge E`` sets it, ``--faces N`` fixes the budget
   instead, ``--no_thinning`` keeps every mesh as it is; ``--isotropic``
   remeshes every piece isotropically to that edge length instead -- the
   triangulation a model trained with ``--remesh_edge`` sees. An object within a
   factor of ``--tolerance`` (2) of the density is left alone, and so is one
   whose pieces share their breaks (at least 1% of their vertices in a
   coincidence cluster: simulated sets such as Fractura's synthetic bones):
   thinning each piece on its own would pull the two sides of every break
   apart, and the model's matches rest on them.

Each scene is one folder, ``<out>/<subset>/<category>/<object>/fractured_0``;
``<out>/<subset>.csv`` records, per assembly, what was done and the faces the
model's dihedral test flags before and after, against the break faces the
dataset marks where it marks them.

Where the dataset marks its breaks (Fractura's ``shared_faces``, Fantastic
Breaks' ``mask``), the vertices on them are written beside the scene too
(``fractured_0/break_vertices.npy``; a piece with no marks of its own, Fantastic
Breaks' restoration, takes the vertices within 0.03 of the other piece's), so
the evaluation can be run with the true breaks in place of the dihedral test --
the oracle experiment::

    python -m scripts.train --root_dir external --evaluate --checkpoint path/to/best.pt \\
        --data_subsets bone_real --split all --modes_per_scene 0 --max_fragments 20 \\
        --label_method marked --checkpoint_dir eval/bone_real_oracle

If the matches improve a great deal, the fracture labels are what fails on
scans; if they stay at chance, the embedding is. Scenes converted before this
file existed have to be converted again for it.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reassembly.data.external import (Preparation, SCALES, fantastic_breaks_entries,  # noqa: E402
                                      garf_entries, prepare, read_fantastic_breaks,
                                      read_garf, reference_edge, write)
from reassembly.mesh.decimate import TOLERANCE  # noqa: E402

# Measured for the error message only: seven Breaking Bad scenes' median edge
# over the object's largest side spans 0.0064 to 0.0230 (median 0.0145).
_MEASURED = "seven Breaking Bad scenes measured in this project: 0.0064 to 0.0230, median 0.0145"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+",
                        help="a GARF-format .hdf5 file (Fractura) or a Fantastic Breaks folder")
    parser.add_argument("--out", required=True, help="the converted data's root folder")
    parser.add_argument("--subsets", nargs="+", default=None,
                        help="a subset name per source (default: the file's or folder's "
                             "name, e.g. bone_real)")
    parser.add_argument("--split", default="all", choices=["all", "train", "val"],
                        help="HDF5 only: the assemblies its split lists give (default all)")
    density = parser.add_mutually_exclusive_group()
    density.add_argument("--reference", default=None, metavar="DATA",
                         help="measure the density on these Breaking Bad scenes "
                              "(the --root_dir you train with)")
    density.add_argument("--edge", type=float, default=None,
                         help="the target median edge as a fraction of the object's "
                              "largest side")
    density.add_argument("--faces", type=int, default=None,
                         help="a fixed face budget per object instead, split by area")
    density.add_argument("--no_thinning", action="store_true",
                         help="keep every mesh at its own density")
    parser.add_argument("--isotropic", action="store_true",
                        help="with --reference or --edge: remesh every piece isotropically "
                             "to that edge length instead of thinning it -- the triangulation "
                             "a model trained with --remesh_edge sees. Every piece, sharing "
                             "its breaks or not. Needs pymeshlab (on Linux also libopengl0)")
    parser.add_argument("--reference_subsets", nargs="+", default=["everyday_compressed"],
                        help="with --reference: the subsets measured (default "
                             "everyday_compressed, what the model was trained on)")
    parser.add_argument("--reference_scenes", type=int, default=200,
                        help="with --reference: how many scenes, spread over the catalogue")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE,
                        help=f"thin only objects denser than the target by more than "
                             f"this factor in edge length (default {TOLERANCE:g})")
    parser.add_argument("--scale", default="object", choices=SCALES,
                        help="object (default): largest side 1, Breaking Bad's convention; "
                             "piece: GARF's loader, scaled down until the largest piece is "
                             "at most 1")
    parser.add_argument("--limit", type=int, default=0,
                        help="convert only the first N assemblies of each source (0: all)")
    parser.add_argument("--workers", type=int, default=1, help="processes (default 1)")
    return parser


def _subset_name(source: Path) -> str:
    name = source.stem if source.is_file() else source.name
    return name.replace("-", "_")       # data.paths.base_subset splits on "-"


def _tasks(source: Path, split: str, limit: int):
    if source.is_file():
        entries = [("hdf5", str(source), entry, split_name)
                   for entry, split_name in garf_entries(source, split)]
    elif source.is_dir():
        entries = [("fantastic_breaks", str(source), entry, k)
                   for entry, k in fantastic_breaks_entries(source)]
    else:
        raise SystemExit(f"{source}: no such file or folder")
    return entries[:limit] if limit else entries


def _convert(task, out: str, subset: str, settings: Preparation) -> dict:
    kind, source, entry, extra = task
    started = time.perf_counter()
    try:
        assembly = (read_garf(source, entry, extra) if kind == "hdf5"
                    else read_fantastic_breaks(source, entry, extra))
        pieces, row, breaks = prepare(assembly, settings, return_breaks=True)
        if pieces is not None:
            row["folder"] = write(pieces, Path(out) / subset / row["scene"],
                                  breaks=breaks).as_posix()
    except Exception as error:                      # noqa: BLE001 -- named in the report
        row = dict(entry=entry, status=f"failed: {type(error).__name__}: {error}")
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row


def _summary(subset: str, rows: list) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    lines = [f"{subset}: {len(ok)} of {len(rows)} assemblies converted"]
    reasons = {}
    for r in rows:
        if r.get("status") != "ok":
            reasons[r["status"]] = reasons.get(r["status"], 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
        lines.append(f"  {count} {reason}")
    if ok:
        def middle(key, whole=False):
            values = [float(r[key]) for r in ok if r.get(key) not in (None, "")
                      and not math.isnan(float(r[key]))]
            if not values:
                return "-"
            value = statistics.median(values)
            return f"{value:,.0f}" if whole else f"{value:.4g}"
        lines += [
            f"  faces per object (median): stored {middle('faces_stored', True)}, cleaned "
            f"{middle('faces_clean', True)}, written {middle('faces', True)}; thinned "
            f"{sum(bool(r['thinned']) for r in ok)} of {len(ok)}",
            f"  median edge / largest side (median): {middle('edge_before')} -> "
            f"{middle('edge_after')}",
            f"  merged vertices {sum(int(r['merged_vertices']) for r in ok):,}, interior and "
            f"repeated faces removed {sum(int(r['duplicate_faces']) for r in ok):,}, debris "
            f"components dropped {sum(int(r['dropped_components']) for r in ok)}",
            f"  copies of another piece removed from a piece: "
            f"{sum(int(r['copies_removed']) for r in ok)} "
            f"({sum(int(r['copy_faces']) for r in ok):,} faces)",
            f"  edges left with one face {sum(int(r['boundary_edges']) for r in ok):,}, with "
            f"more than two {sum(int(r['nonmanifold_edges']) for r in ok):,} (in "
            f"{sum(int(r['nonmanifold_edges']) > 0 for r in ok)} of {len(ok)} objects)",
            f"  objects whose pieces share vertices (so with an embedding loss): "
            f"{sum(int(r['shared_vertices']) > 0 for r in ok)} of {len(ok)}; not thinned "
            f"because their pieces share their breaks: "
            f"{sum(str(r.get('shares_breaks')) == 'True' for r in ok)}",
            f"  faces the dihedral test flags (median share): {middle('flagged_full')} before "
            f"thinning, {middle('flagged')} after",
        ]
        failed = sum(int(r.get("isotropic_failed") or 0) for r in ok)
        if any(r.get("isotropic") for r in ok):
            lines.append(f"  remeshed isotropically; pieces the remesher could not work on, kept "
                         f"as cleaned: {failed}")
        marked = [r for r in ok if not math.isnan(float(r["break_share"]))]
        if marked and any(float(r["break_share"]) > 0 for r in marked):
            lines.append(f"  against the dataset's break faces ({middle('break_share')} of "
                         f"faces): recall {middle('recall_full')} -> {middle('recall')}, "
                         f"precision {middle('precision_full')} -> {middle('precision')}")
        elif marked:
            lines.append("  the dataset marks no break face on these pieces")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.reference or args.edge or args.faces or args.no_thinning):
        raise SystemExit("choose the density: --reference DATA (measures your Breaking Bad "
                         "scenes), --edge E, --faces N or --no_thinning. For scale, "
                         f"{_MEASURED}.")
    if args.isotropic and not (args.reference or args.edge):
        raise SystemExit("--isotropic remeshes to an edge length: give --reference DATA or "
                         "--edge E with it")
    if args.isotropic:
        from reassembly.data.remesh import _pymeshlab

        try:                     # checked now, not after cleaning every assembly
            _pymeshlab()
        except ImportError as error:
            raise SystemExit(str(error)) from None
    elif not args.no_thinning:
        try:                     # checked now, not after cleaning every assembly
            import fast_simplification  # noqa: F401
        except ImportError:
            raise SystemExit("thinning needs fast-simplification: pip install "
                             "fast-simplification (or --no_thinning)") from None
    sources = [Path(s) for s in args.sources]
    subsets = args.subsets or [_subset_name(s) for s in sources]
    if len(subsets) != len(sources):
        raise SystemExit("--subsets: one name per source")

    edge = args.edge
    if args.reference:
        edge, count, (low, high) = reference_edge(args.reference, args.reference_subsets,
                                                  args.reference_scenes)
        print(f"Breaking Bad density, {count} scenes of {', '.join(args.reference_subsets)}: "
              f"median edge {edge:.4f} of the object's largest side "
              f"(quartiles {low:.4f}, {high:.4f})")
    settings = Preparation(edge=edge, faces=args.faces, tolerance=args.tolerance,
                           scale=args.scale, isotropic=args.isotropic)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for source, subset in zip(sources, subsets):
        tasks = _tasks(source, args.split, args.limit)
        print(f"\n{source} -> {out / subset}: {len(tasks)} assemblies")
        rows = []
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                futures = [pool.submit(_convert, t, str(out), subset, settings) for t in tasks]
                for index, future in enumerate(futures, 1):
                    rows.append(future.result())
                    print(f"\r  {index}/{len(tasks)}", end="", flush=True)
        else:
            for index, task in enumerate(tasks, 1):
                rows.append(_convert(task, str(out), subset, settings))
                print(f"\r  {index}/{len(tasks)}", end="", flush=True)
        print()
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with open(out / f"{subset}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(_summary(subset, rows))
        print(f"  report: {out / f'{subset}.csv'}")
        print(f"  evaluate: python -m scripts.train --root_dir {out} --evaluate --checkpoint "
              f"path/to/best.pt --data_subsets {subset} --split all --modes_per_scene 0 "
              f"--max_fragments 20 --checkpoint_dir eval/{subset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

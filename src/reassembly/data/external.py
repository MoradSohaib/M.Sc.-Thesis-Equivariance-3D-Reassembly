"""
Other datasets in Breaking Bad's place: Fractura (GARF's HDF5) and Fantastic
Breaks.

Each is read as assemblies -- every piece's mesh in the assembled pose, and the
faces the dataset marks as break where it marks any -- and prepared the way the
training data already is: cleaned to one closed surface per piece
(:mod:`reassembly.mesh.clean`), scaled so the object's largest side is 1, as
Breaking Bad's objects are, and thinned to Breaking Bad's density
(:mod:`reassembly.mesh.decimate`). ``scripts/convert_external.py`` writes the
result in Breaking Bad's layout (:func:`reassembly.data.scene.write_scene`), so
``--evaluate``, the figures and the Visualizer run on it with no other change,
and the network computes its normals and fracture labels exactly as on Breaking
Bad -- from the vertices and faces, which are all the readers here, and
Breaking Bad's, pass it.

Fractura, as GARF writes it (``scripts/process_breakingbad.py`` in GARF's
repository)::

    data_split/<category>/{train,val}     entry names
    <entry>/pieces/<i>/vertices, faces    one mesh per piece, i = 0, 1, ...
    <entry>/pieces/<i>/shared_faces       per face: the piece it shares a
                                          break with, -1 for none

Fantastic Breaks::

    <group>/<object>/model_b_<k>.ply      the broken object
    <group>/<object>/model_r_<k>.ply      the restoration: what broke off
    <group>/<object>/meta_<k>.npz         'mask', the break on model_b, per
                                          vertex or per face
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..mesh.clean import MERGE_TOLERANCE, clean_piece, copies_of_pieces, edge_counts, \
    identical_pieces
from ..mesh.decimate import TOLERANCE, ThinReport, median_edge, object_extent, thin_assembly, \
    transfer_face_labels

SCALES = ("object", "piece")
MODE = "fractured_0"
_MODE_PREFIXES = ("fractured", "mode")
# An assembly's pieces share their breaks when at least this share of their
# vertices lies in a coincidence cluster, and then it is never thinned: each
# piece is thinned on its own, so the two sides of a break stop coinciding --
# on the four synthetic objects first converted that way, 97.8% of the shared
# vertices went, and part accuracy fell from 0.66 to 0.05 over their 29 scenes.
# Measured at full resolution: 3.2% to 56% in 91 of Fractura's 93 synthetic
# scenes (none in the other two) and 4.4% to 54% in seven Breaking Bad scenes,
# each coincident face facing its twin; none in the 36 real Fractura objects
# checked once vert8's stored copy is gone (copies_of_pieces). Fantastic
# Breaks was not measured at full resolution.
SHARED_BREAKS = 0.01
# A piece the dataset marks no break on -- Fantastic Breaks' restoration, whose
# meta file marks the broken object only -- takes as its break the vertices
# within this distance (an object's largest side being 1) of another piece's
# marked ones. Fractura's real pieces, converted, stand 0.011 to 0.029 apart
# where they face each other (median 0.0184), so this reaches across the gap.
BREAK_REACH = 0.03
# Passes of the isotropic remesher when converting (Preparation.isotropic): once
# per object, so the two more than training's three are cheap.
ISOTROPIC_PASSES = 5


@dataclass
class Assembly:
    """One broken object as a dataset stores it."""

    source: str                      # the file or folder it was read from
    entry: str                       # its name there
    category: str                    # folders between subset and object
    name: str                        # the object's folder
    split: str                       # as the dataset lists it; "" when it does not
    pieces: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    breaks: List[Optional[np.ndarray]] = field(default_factory=list)   # per face


@dataclass(frozen=True)
class Preparation:
    """How :func:`prepare` turns an assembly into the model's input."""

    edge: Optional[float] = None     # target median edge / largest extent
    faces: Optional[int] = None      # or a fixed face budget per object
    tolerance: float = TOLERANCE     # thin only when denser by more than this
    scale: str = "object"            # "object": largest side 1 (Breaking Bad);
                                     # "piece": GARF's loader, largest piece at most 1
    merge_tolerance: float = MERGE_TOLERANCE
    isotropic: bool = False          # remesh every piece isotropically to `edge`
                                     # instead of thinning it

    def __post_init__(self):
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}, got {self.scale!r}")
        if self.edge is not None and self.faces is not None:
            raise ValueError("give edge or faces, not both")
        if self.isotropic and self.edge is None:
            raise ValueError("isotropic remeshing needs the edge length to remesh to (edge)")


# ---------------------------------------------------------------- reading --

def place(entry: str) -> Tuple[str, str]:
    """
    ``(category, object folder)`` for an entry path: ``pig_bone/pig_bone_1``
    -> ``("pig_bone", "pig_bone_1")``. A trailing fracture-mode component is
    folded into the object, ``pig/synthetic_fracture/3/mode_1`` ->
    ``("pig/synthetic_fracture", "3_mode_1")``: each pattern is its own mesh,
    and a scene folder holds one mesh.
    """
    parts = [p for p in entry.replace("\\", "/").split("/") if p]
    if len(parts) >= 2 and parts[-1].startswith(_MODE_PREFIXES):
        return "/".join(parts[:-2]), f"{parts[-2]}_{parts[-1]}"
    return "/".join(parts[:-1]), parts[-1]


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def garf_entries(path, split: str = "all") -> List[Tuple[str, str]]:
    """
    ``(entry, split)`` for every assembly in a GARF-format HDF5, in the order
    its split lists give them; ``split`` keeps one list. A file without lists:
    every group holding ``pieces``, with no split.
    """
    import h5py

    entries: Dict[str, str] = {}
    with h5py.File(path, "r") as f:
        if "data_split" in f:
            for category in f["data_split"].values():
                for name, listed in category.items():
                    if split not in ("all", name):
                        continue
                    for entry in np.atleast_1d(listed[()]):
                        if isinstance(entry, (bytes, str)):
                            entries.setdefault(_text(entry).strip("/"), name)
        else:
            def visit(name, obj):
                if isinstance(obj, h5py.Group) and "pieces" in obj:
                    entries.setdefault(name, "")
            f.visititems(visit)
    return list(entries.items())


def read_garf(path, entry: str, split: str = "") -> Assembly:
    """One assembly of a GARF-format HDF5, pieces in numeric order."""
    import h5py

    category, name = place(entry)
    assembly = Assembly(str(path), entry, category, name, split)
    with h5py.File(path, "r") as f:
        pieces = f[entry]["pieces"]
        for key in sorted(pieces, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else k)):
            group = pieces[key]
            vertices = np.asarray(group["vertices"][()], dtype=np.float64)
            faces = np.asarray(group["faces"][()], dtype=np.int64)
            breaks = None
            if "shared_faces" in group:
                shared = np.asarray(group["shared_faces"][()])
                breaks = shared if shared.dtype == bool else shared >= 0
                if len(breaks) != len(faces):
                    breaks = None
            assembly.pieces.append((vertices, faces))
            assembly.breaks.append(breaks)
    return assembly


def fantastic_breaks_entries(root) -> List[Tuple[str, int]]:
    """``(object folder relative to root, k)`` for every model_b_<k> with a model_r_<k>."""
    root = Path(root)
    found = []
    for broken in sorted(root.glob("*/*/model_b_*.ply")) + sorted(root.glob("*/model_b_*.ply")):
        k = broken.stem[len("model_b_"):]
        if k.isdigit() and (broken.parent / f"model_r_{k}.ply").is_file():
            found.append((broken.parent.relative_to(root).as_posix(), int(k)))
    return list(dict.fromkeys(found))


def _load_ply(path) -> Tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(path, process=False, force="mesh")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def read_fantastic_breaks(root, entry: str, k: int = 0) -> Assembly:
    """The broken object and its restoration, break faces on the first from ``meta_<k>``."""
    folder = Path(root) / entry
    category, name = place(entry)
    if k:
        name = f"{name}_{k}"
    assembly = Assembly(str(root), f"{entry}/{k}", category, name, "")
    broken = _load_ply(folder / f"model_b_{k}.ply")
    restoration = _load_ply(folder / f"model_r_{k}.ply")
    breaks = None
    meta = folder / f"meta_{k}.npz"
    if meta.is_file():
        try:
            with np.load(meta, allow_pickle=False) as data:
                mask = np.asarray(data["mask"]).ravel() if "mask" in data.files else None
        except ValueError:                       # object arrays: not read
            mask = None
        if mask is not None:
            mask = mask.astype(bool)
            if len(mask) == len(broken[0]):
                breaks = mask[broken[1]].all(axis=1)
            elif len(mask) == len(broken[1]):
                breaks = mask
    assembly.pieces = [broken, restoration]
    assembly.breaks = [breaks, None]
    return assembly


# --------------------------------------------------------------- preparing --

def shared_vertices(pieces: Sequence) -> int:
    """
    Vertices in a coincidence cluster -- the ones the embedding loss pairs
    across pieces -- found exactly as training finds them
    (:func:`~reassembly.mesh.correspondence.find_shared_points`, its defaults).
    """
    from ..mesh.correspondence import find_shared_points

    return sum(len(cluster) for cluster in find_shared_points([v for v, _ in pieces]))


def shares_breaks(pieces: Sequence) -> Tuple[int, bool]:
    """``(shared vertices, whether they are at least SHARED_BREAKS of all the vertices)``."""
    count = shared_vertices(pieces)
    return count, bool(count >= SHARED_BREAKS * sum(len(v) for v, _ in pieces))


def _flagged(pieces: Sequence, threshold: float = 0.9) -> List[np.ndarray]:
    """The dihedral fracture faces -- the labels the model's checkpoint uses at test time."""
    from ..mesh.fracture import fracture_face_mask

    return [fracture_face_mask(v, f, sharp_threshold=threshold, min_faces=1).face_mask
            for v, f in pieces]


def _agreement(flagged: Sequence, breaks: Sequence) -> Tuple[float, float, float]:
    """``(break share, precision, recall)`` of the flagged faces against the marked ones."""
    pairs = [(m, b) for m, b in zip(flagged, breaks) if b is not None and len(b) == len(m)]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    hit = sum(int((m & b).sum()) for m, b in pairs)
    marked = sum(int(b.sum()) for _, b in pairs)
    chosen = sum(int(m.sum()) for m, _ in pairs)
    total = sum(len(b) for _, b in pairs)
    if not marked:                       # nothing marked: nothing to agree with
        return 0.0, float("nan"), float("nan")
    return (marked / total, hit / chosen if chosen else float("nan"), hit / marked)


def _without_debris(piece, merge_tolerance: float):
    """
    ``piece`` cleaned again, its debris dropped. A piece of one component has
    none, and the pass would change nothing: it comes back as it is, its
    report saying that nothing more was done.
    """
    if piece.report.components <= 1:
        return piece._replace(report=replace(piece.report, merged_vertices=0,
                                             collapsed_faces=0, duplicate_faces=0,
                                             dropped=(), flipped=False))
    return clean_piece(piece.vertices, piece.faces, piece.face_labels,
                       merge_tolerance=merge_tolerance)


def vertex_breaks(pieces: Sequence, breaks: Sequence,
                  reach: float = BREAK_REACH) -> Optional[List[np.ndarray]]:
    """
    Per piece, which vertices lie on a break the dataset marks: the corners of
    its marked faces (``breaks``, per face, or ``None``). A piece with no
    marks takes the vertices within ``reach`` of another piece's marked ones
    (:data:`BREAK_REACH`). ``None`` when no piece is marked at all.
    """
    from scipy.spatial import cKDTree

    masks: List[Optional[np.ndarray]] = []
    for (vertices, faces), marked in zip(pieces, breaks):
        faces = np.asarray(faces)
        if marked is None or len(marked) != len(faces):
            masks.append(None)
            continue
        mask = np.zeros(len(vertices), dtype=bool)
        mask[faces[np.asarray(marked, dtype=bool)].ravel()] = True
        masks.append(mask)
    if all(mask is None for mask in masks):
        return None
    points = [np.asarray(v, dtype=np.float64)[m] for (v, _), m in zip(pieces, masks)
              if m is not None]
    points = np.concatenate(points) if points else np.zeros((0, 3))
    tree = cKDTree(points) if len(points) else None
    out = []
    for (vertices, _), mask in zip(pieces, masks):
        if mask is None:
            if tree is None:
                mask = np.zeros(len(vertices), dtype=bool)
            else:
                distance, _ = tree.query(np.asarray(vertices, dtype=np.float64),
                                         distance_upper_bound=reach)
                mask = np.isfinite(distance)
        out.append(mask)
    return out


def prepare(assembly: Assembly, settings: Preparation = Preparation(),
            return_breaks: bool = False):
    """
    ``(pieces, row)``: the assembly cleaned, scaled and thinned, and what was
    done to it; ``pieces`` is ``None`` when it cannot be used (``row["status"]``
    says why) -- fewer than two pieces left, or two pieces that are one mesh
    stored twice. A piece stored again inside another is removed from it
    (:func:`~reassembly.mesh.clean.copies_of_pieces`), and pieces that share
    their breaks (:data:`SHARED_BREAKS`) are not thinned.

    ``return_breaks`` adds a third value: per piece of ``pieces``, which
    vertices lie on the break the dataset marks (:func:`vertex_breaks`), or
    ``None`` when it marks none -- what ``scripts/convert_external.py`` writes
    beside the scene for the oracle evaluation (``--label_method marked``).
    """
    if return_breaks:
        pieces, row, breaks = _prepare(assembly, settings)
        return pieces, row, breaks
    return _prepare(assembly, settings)[:2]


def _prepare(assembly: Assembly, settings: Preparation):
    """:func:`prepare`, with the marked break vertices as a third value."""
    row: Dict[str, object] = dict(
        entry=assembly.entry, scene=f"{assembly.category}/{assembly.name}".strip("/"),
        split=assembly.split, pieces=len(assembly.pieces),
        faces_stored=sum(len(f) for _, f in assembly.pieces))
    # Every component kept first, as the Breaking Bad reader keeps them; the
    # debris goes in a second pass, used only for pieces that do not share
    # their breaks -- on simulated pieces the "debris" is the object's own
    # geometry (105 islands of 4 to 58 faces in 18 of Fractura's synthetic
    # scenes, in three of them one with 8 vertices of a break), on scanned
    # ones it is the scan's (Blender's cube in every real bone piece).
    whole = [clean_piece(vertices, faces, marked, merge_tolerance=settings.merge_tolerance,
                         drop_debris=False)
             for (vertices, faces), marked in zip(assembly.pieces, assembly.breaks)]
    merged = sum(p.report.merged_vertices for p in whole)
    duplicates = sum(p.report.duplicate_faces for p in whole)
    turned = [p.report.flipped for p in whole]
    # A copy of another piece is not this piece's surface: every vertex of it
    # coincides with that piece's, as a break's do, but each face points the
    # same way as its twin instead of against it, and kept, the model would
    # match the piece to its own copy.
    copies = copies_of_pieces([(p.vertices, p.faces) for p in whole],
                              components=[p.report.components for p in whole])
    removed: Dict[int, np.ndarray] = {}
    for i, mask, _ in copies:
        removed[i] = removed.get(i, np.zeros(len(mask), dtype=bool)) | mask
    for i, mask in removed.items():
        piece = whole[i]
        labels = None if piece.face_labels is None else piece.face_labels[~mask]
        whole[i] = clean_piece(piece.vertices, piece.faces[~mask], labels,
                               merge_tolerance=settings.merge_tolerance, drop_debris=False)
        merged += whole[i].report.merged_vertices
        duplicates += whole[i].report.duplicate_faces
        turned[i] ^= whole[i].report.flipped
    tidy = [_without_debris(p, settings.merge_tolerance) for p in whole]
    usable = [(w, t) for w, t in zip(whole, tidy) if len(t.faces)]
    if usable:
        extent = object_extent([(t.vertices, t.faces) for _, t in usable])
        _, keep = shares_breaks([(t.vertices / extent, t.faces) for _, t in usable])
    else:
        keep = False
    chosen = [w if keep else t for w, t in usable]
    row.update(merged_vertices=merged, duplicate_faces=duplicates,
               copies_removed=len(copies), copy_faces=sum(int(m.sum()) for _, m, _ in copies),
               dropped_components=0 if keep else sum(len(t.report.dropped) for t in tidy),
               flipped=sum(was != (t.report.flipped and not keep)
                           for was, t in zip(turned, tidy)),
               faces_clean=sum(len(p.faces) for p in chosen))
    cleaned = [(p.vertices, p.faces) for p in chosen]
    breaks = [p.face_labels for p in chosen]
    if len(cleaned) < 2:
        row["status"] = f"skipped: {len(cleaned)} piece(s) after cleaning"
        return None, row, None
    twins = identical_pieces(cleaned)
    if twins:
        row["status"] = "skipped: pieces " + ", ".join(f"{i} and {j}" for i, j in twins) + \
            " are the same mesh"
        return None, row, None

    extent = object_extent(cleaned)
    if settings.scale == "object":
        scale = 1.0 / extent
    else:                                # GARF's loader: the largest piece to 1, down only
        scale = 1.0 / max(1.0, max(float(np.ptp(v, axis=0).max()) for v, _ in cleaned))
    cleaned = [(v * scale, f) for v, f in cleaned]
    row.update(scale=scale, extent_stored=extent)

    flagged = _flagged(cleaned)
    share, precision, recall = _agreement(flagged, breaks)
    row.update(break_share=share, flagged_full=float(np.mean(np.concatenate(flagged))),
               precision_full=precision, recall_full=recall)

    row.update(shared_full=shared_vertices(cleaned), shares_breaks=keep)
    if settings.isotropic:
        # Every piece, sharing its breaks or not: the point is that the scans
        # and the training scenes (--remesh_edge) are triangulated alike, and a
        # model trained that way does not rest on shared vertices.
        from .remesh import remesh_isotropic

        target = settings.edge * object_extent(cleaned)
        thinned = [remesh_isotropic(v, f, target, iterations=ISOTROPIC_PASSES)
                   for v, f in cleaned]
        before = median_edge(cleaned) / object_extent(cleaned)
        report = ThinReport(True, before, median_edge(thinned) / object_extent(thinned),
                            sum(len(f) for _, f in cleaned), sum(len(f) for _, f in thinned),
                            0)
        row["isotropic"] = settings.edge
    elif keep:                           # no target: measured, returned as it is
        thinned, report = thin_assembly(cleaned)
    else:
        thinned, report = thin_assembly(cleaned, edge=settings.edge, faces=settings.faces,
                                        tolerance=settings.tolerance)
    tidied = 0
    if report.thinned:
        # Edge collapse can leave a face twice or a sliver; the Breaking Bad
        # reader would drop them on loading, so they go here and the file is
        # exactly what the model gets.
        kept = []
        for v, f in thinned:
            # Pieces that share their breaks keep their debris, as cleaned
            # (simulated islands are the object's own geometry).
            piece = clean_piece(v, f, merge_tolerance=settings.merge_tolerance,
                                drop_debris=not keep)
            tidied += len(f) - len(piece.faces)
            kept.append((piece.vertices, piece.faces))
        thinned = kept
        breaks = [None if b is None else transfer_face_labels(v0, f0, b, v1, f1)
                  for (v0, f0), (v1, f1), b in zip(cleaned, thinned, breaks)]
        flagged = _flagged(thinned)
    _, precision, recall = _agreement(flagged, breaks)
    boundary = nonmanifold = 0
    for _, f in thinned:
        b, n = edge_counts(f)
        boundary, nonmanifold = boundary + b, nonmanifold + n
    row.update(thinned=report.thinned, edge_before=report.edge_before,
               edge_after=report.edge_after, faces=sum(len(f) for _, f in thinned),
               tidied_faces=tidied,
               flagged=float(np.mean(np.concatenate(flagged))), precision=precision,
               recall=recall, boundary_edges=boundary, nonmanifold_edges=nonmanifold,
               shared_vertices=shared_vertices(thinned), status="ok")
    marked = vertex_breaks(thinned, breaks)
    row.update(marked_pieces=sum(b is not None and len(b) == len(f)
                                 for (_, f), b in zip(thinned, breaks)),
               marked_vertices=float(np.mean(np.concatenate(marked))) if marked else float("nan"))
    return thinned, row, marked


def write(pieces: Sequence, scene_dir, breaks: Optional[Sequence[np.ndarray]] = None) -> Path:
    """
    The pieces as one Breaking Bad scene folder, replacing what was there;
    ``breaks``, per piece the vertices on a marked break (:func:`prepare`),
    written beside it (:data:`reassembly.data.scene.BREAK_FILE`).
    """
    from .scene import write_scene

    scene_dir = Path(scene_dir)
    partial = scene_dir.with_name(scene_dir.name + ".part")
    if partial.exists():
        shutil.rmtree(partial)
    write_scene(partial, pieces, MODE, breaks=breaks)
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    os.replace(partial, scene_dir)
    return scene_dir


# -------------------------------------------------------------- reference --

def reference_edge(root, subsets: Optional[Sequence[str]] = None,
                   scenes: int = 200) -> Tuple[float, int, Tuple[float, float]]:
    """
    The density to thin to, measured on Breaking Bad itself: the median, over
    up to ``scenes`` scenes spread across the catalogue (the first ``fractured_``
    mode of each), of the median edge over the object's largest side. Returns
    ``(median, scenes measured, (lower quartile, upper quartile))``.
    """
    from .paths import find_scenes
    from .scene import SceneReader

    found = find_scenes(root, subsets)
    if not found:
        raise FileNotFoundError(f"no Breaking Bad scenes under {root}"
                                + (f" in {list(subsets)}" if subsets else ""))
    picks = sorted(set(np.linspace(0, len(found) - 1, min(scenes, len(found)))
                       .round().astype(int).tolist()))
    values = []
    for index in picks:
        reader = SceneReader(found[index].path)
        modes = [m for m in reader.mode_names() if m.startswith("fractured_")]
        if not modes:
            continue
        pieces = [(m.vertices, m.faces) for m in reader.load_mode(modes[0]).fragments]
        if len(pieces) < 2:
            continue
        extent = object_extent(pieces)
        if extent > 0:
            values.append(median_edge(pieces) / extent)
    if not values:
        raise ValueError(f"no measurable scene under {root}")
    values = np.asarray(values)
    return (float(np.median(values)), len(values),
            (float(np.percentile(values, 25)), float(np.percentile(values, 75))))

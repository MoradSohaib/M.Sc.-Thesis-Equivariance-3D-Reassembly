"""
Fractura and Fantastic Breaks in Breaking Bad's place (``data/external.py``,
``mesh/clean.py``, ``mesh/decimate.py``, ``scripts/convert_external.py``).

The fixtures rebuild, on a small tube cut into pieces, each defect found in the
real files: vertices repeated per face and slightly apart (Fractura's real
bones), Blender's default cube at the origin in every piece (the same), doubled
interior walls (Fractura's synthetic pieces), millimetre units, and a piece
stored twice. Each of them still produces a number when left in -- a soup of
loose triangles is a valid mesh -- which is why each is tested here.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from reassembly.data.external import (Assembly, Preparation, fantastic_breaks_entries,  # noqa: E402
                                      garf_entries, place, prepare, read_fantastic_breaks,
                                      read_garf, reference_edge, write)
from reassembly.data.paths import find_scenes  # noqa: E402
from reassembly.data.scene import SceneReader, write_scene  # noqa: E402
from reassembly.mesh.clean import (clean_piece, edge_counts, face_components,  # noqa: E402
                                   identical_pieces)
from reassembly.mesh.decimate import (median_edge, object_extent, surface_area,  # noqa: E402
                                      thin_assembly, transfer_face_labels)

CUBE_VERTICES = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
CUBE_FACES = trimesh.creation.box(extents=(2, 2, 2)).faces    # 12 triangles


def _rings(z, sections):
    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    radius = 1.0 + 0.5 * (z / 3.0) ** 4                       # wider at the ends, like a bone
    return np.stack([np.outer(radius, np.cos(angles)), np.outer(radius, np.sin(angles)),
                     np.repeat(z[:, None], sections, 1)], -1).reshape(-1, 3)


def tube_pieces(cuts, sections=24, rings=40, walls=()):
    """
    A closed tube along z cut at ``cuts``: each piece's side wall and two
    capping fans, neighbours' caps at identical points facing opposite ways
    (Breaking Bad's coincident breaks). ``walls``: heights inside piece 0
    where it gets an interior disk stored twice, once each way. Returns
    ``(pieces, breaks)``, breaks per face: the neighbour across, -1 none.
    """
    heights = np.linspace(-3, 3, rings)
    edges = [-3.0, *cuts, 3.0]
    pieces, breaks = [], []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        inside = [w for w in walls if low < w < high] if index == 0 else []
        z = np.unique(np.concatenate([[low, high], inside,
                                      heights[(heights > low) & (heights < high)]]))
        vertices = list(_rings(z, sections))
        faces, partner = [], []
        i = np.arange(sections)
        j = (i + 1) % sections
        for r in range(len(z) - 1):
            a, b = r * sections + i, r * sections + j
            faces += [np.stack([a, b, b + sections], 1), np.stack([a, b + sections, a + sections], 1)]
            partner += [-1] * (2 * sections)
        for level, sign, neighbour in ((0, -1, index - 1), (len(z) - 1, 1, index + 1)):
            centre = len(vertices)
            vertices.append([0.0, 0.0, z[level]])
            ring = level * sections
            fan = (np.stack([np.full(sections, centre), ring + j, ring + i], 1) if sign < 0 else
                   np.stack([np.full(sections, centre), ring + i, ring + j], 1))
            faces.append(fan)
            across = neighbour if 0 <= neighbour < len(edges) - 1 else -1
            partner += [across] * sections
        for height in inside:                                   # interior disk, twice
            level = int(np.flatnonzero(np.isclose(z, height))[0])
            centre = len(vertices)
            vertices.append([0.0, 0.0, height])
            ring = level * sections
            disk = np.stack([np.full(sections, centre), ring + i, ring + j], 1)
            faces += [disk, disk[:, ::-1]]
            partner += [-1] * (2 * sections)
        pieces.append((np.asarray(vertices, float), np.concatenate(faces).astype(np.int64)))
        breaks.append(np.asarray(partner, np.int64))
    return pieces, breaks


def as_scan(vertices, faces, rng, scale=30.0, cube=True):
    """What the real bones store: millimetres, one vertex copy per face corner,
    the copies a little apart, and the default cube at the origin."""
    corners = vertices[faces].reshape(-1, 3) * scale
    diagonal = np.linalg.norm(np.ptp(corners, axis=0))
    corners = corners + rng.standard_normal(corners.shape) * 3e-8 * diagonal
    soup = np.arange(len(corners)).reshape(-1, 3)
    if not cube:
        return corners, soup
    return (np.vstack([corners, CUBE_VERTICES]),
            np.vstack([soup, CUBE_FACES + len(corners)]))


def write_garf(path, entries, split="val"):
    """``entries``: name -> (pieces, breaks), in GARF's layout."""
    h5py = pytest.importorskip("h5py")              # only the HDF5 tests need it
    with h5py.File(path, "w") as f:
        group = f.create_group("data_split").create_group(next(iter(entries)).split("/")[0])
        group.create_dataset("train", data=np.zeros(0))
        group.create_dataset(split, data=np.array(list(entries), dtype=object),
                             dtype=h5py.string_dtype())
        for name, (pieces, breaks) in entries.items():
            node = f.create_group(name)
            node.create_dataset("pieces_names", data=np.array(
                [f"piece_{i}" for i in range(len(pieces))], dtype=object), dtype=h5py.string_dtype())
            for i, ((v, faces), b) in enumerate(zip(pieces, breaks)):
                piece = node.create_group(f"pieces/{i}")
                piece.create_dataset("vertices", data=v)
                piece.create_dataset("faces", data=faces)
                piece.create_dataset("shared_faces", data=b)
    return path


@pytest.fixture
def thinning():
    """The tests that thin need fast-simplification; without it they skip."""
    pytest.importorskip("fast_simplification")


@pytest.fixture
def real_like():
    """Two tube pieces stored as the real bones are, finer than Breaking Bad."""
    rng = np.random.default_rng(0)
    pieces, breaks = tube_pieces([0.4], sections=48, rings=80)
    stored, marked = [], []
    for (v, f), b in zip(pieces, breaks):
        sv, sf = as_scan(v, f, rng)
        stored.append((sv, sf))
        marked.append(np.concatenate([b, np.full(len(CUBE_FACES), -1)]))
    return stored, marked


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def test_a_scan_soup_becomes_one_closed_surface_and_the_cube_goes(real_like):
    (vertices, faces), marked = real_like[0][0], real_like[1][0]
    assert face_components(faces, len(vertices))[0] == len(faces) - 11  # loose triangles + cube
    piece = clean_piece(vertices, faces, marked >= 0)
    assert piece.report.dropped and piece.report.dropped[0][0] == 12   # the cube
    assert piece.report.components == 1
    assert edge_counts(piece.faces) == (0, 0)                           # closed, manifold
    assert len(piece.vertices) < len(vertices) / 2
    assert not np.any(np.all(np.abs(piece.vertices) <= 1.0 + 1e-9, axis=1))
    # The break labels travel with their faces: exactly the cap piece 1 shares.
    on_cap = np.all(np.abs(piece.vertices[piece.faces][:, :, 2] - 0.4 * 30) < 1e-3, axis=1)
    assert np.array_equal(piece.face_labels, on_cap)


def test_on_a_small_piece_the_cube_goes_by_its_face_count():
    rng = np.random.default_rng(1)
    pieces, _ = tube_pieces([0.4])
    for scale in (5.0, 0.5):          # the cube 3% of the piece's area, then larger than it
        vertices, faces = as_scan(*pieces[1], rng, scale=scale)
        piece = clean_piece(vertices, faces)
        (count, share), = piece.report.dropped
        assert count == 12 and share > 0.01       # above the area rule: the count drops it
        assert piece.report.components == 1 and edge_counts(piece.faces) == (0, 0)
        assert len(piece.faces) == len(pieces[1][1])


def test_a_small_separate_part_goes_and_a_large_one_stays():
    pieces, _ = tube_pieces([0.4])
    vertices, faces = pieces[1]
    ball = trimesh.creation.icosphere(subdivisions=1)            # 80 faces, more than debris
    for radius, kept in ((0.1, False), (0.5, True)):             # 0.4% and 10% of the area
        extra = np.asarray(ball.vertices) * radius + [5.0, 0.0, 0.0]
        piece = clean_piece(np.vstack([vertices, extra]),
                            np.vstack([faces, np.asarray(ball.faces) + len(vertices)]))
        assert piece.report.components == (2 if kept else 1)
        assert len(piece.faces) == len(faces) + (len(ball.faces) if kept else 0)


def test_vertices_further_apart_than_the_tolerance_stay_apart():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1e-3, 0, 0]], float)
    f = np.array([[0, 1, 2], [3, 1, 2]])
    piece = clean_piece(v, f, merge_tolerance=1e-6)
    assert len(piece.vertices) == 4 and piece.report.merged_vertices == 0
    piece = clean_piece(v, f, merge_tolerance=1e-2)                     # now within it
    assert piece.report.merged_vertices == 1 and piece.report.duplicate_faces == 1


def test_doubled_interior_walls_cancel_and_leave_one_closed_surface():
    pieces, breaks = tube_pieces([0.4], walls=(-1.5, -0.5))
    vertices, faces = pieces[0]
    assert edge_counts(faces)[1] > 0                                    # the walls' rims
    piece = clean_piece(vertices, faces, breaks[0])
    assert piece.report.duplicate_faces == 4 * 24                       # 2 walls x 2 copies
    assert edge_counts(piece.faces) == (0, 0) and piece.report.components == 1
    on_cap = np.all(np.isclose(piece.vertices[piece.faces][:, :, 2], 0.4), axis=1)
    assert np.array_equal(piece.face_labels >= 0, on_cap) and on_cap.sum() == 24


def test_a_surface_wound_inward_is_turned_outward():
    pieces, _ = tube_pieces([0.4])
    vertices, faces = pieces[0]
    piece = clean_piece(vertices, faces[:, ::-1])
    assert piece.report.flipped
    corners = piece.vertices[piece.faces] - piece.vertices.mean(axis=0)
    assert np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() > 0
    assert not clean_piece(vertices, faces).report.flipped


def test_a_piece_stored_twice_is_found():
    pieces, _ = tube_pieces([0.4])
    assert identical_pieces(pieces) == []
    shuffled = pieces[0][0][::-1].copy()
    assert identical_pieces([pieces[0], (shuffled, pieces[0][1]), pieces[1]]) == [(0, 1)]


# --------------------------------------------------------------------------
# Thinning
# --------------------------------------------------------------------------

def _dense(sections=96, rings=160, scale=30.0):
    pieces, breaks = tube_pieces([0.4], sections=sections, rings=rings)
    return [(v * scale, f) for v, f in pieces], breaks


@pytest.mark.usefixtures("thinning")
def test_thinning_reaches_the_density_and_keeps_the_shape():
    pieces, _ = _dense()
    extent = object_extent(pieces)
    target = 4 * median_edge(pieces) / extent
    thinned, report = thin_assembly(pieces, edge=target)
    assert report.thinned and report.faces_after < report.faces_before / 8
    assert abs(report.edge_after / target - 1) < 0.1
    for (v0, f0), (v1, f1) in zip(pieces, thinned):
        assert np.allclose(np.ptp(v1, axis=0), np.ptp(v0, axis=0), rtol=0.02)
        assert abs(surface_area(v1, f1) / surface_area(v0, f0) - 1) < 0.02
    # Area-weighted: the face counts follow the areas, not the input counts.
    areas = [surface_area(v, f) for v, f in pieces]
    counts = [len(f) for _, f in thinned]
    assert abs(counts[0] / counts[1] - areas[0] / areas[1]) < 0.1 * areas[0] / areas[1]


@pytest.mark.usefixtures("thinning")
def test_a_mesh_near_the_density_is_left_alone():
    pieces, _ = tube_pieces([0.4])
    extent = object_extent(pieces)
    edge = median_edge(pieces) / extent
    same, report = thin_assembly(pieces, edge=1.5 * edge)              # within a factor of 2
    assert not report.thinned
    assert all(a[1] is b[1] or np.array_equal(a[1], b[1]) for a, b in zip(pieces, same))
    _, report = thin_assembly(pieces, edge=3 * edge)
    assert report.thinned
    _, report = thin_assembly(pieces)                                   # no target
    assert not report.thinned


@pytest.mark.usefixtures("thinning")
def test_a_fixed_budget_is_split_by_area():
    pieces, _ = _dense()
    thinned, report = thin_assembly(pieces, faces=2000, tolerance=1.0)
    assert report.thinned and abs(report.faces_after - 2000) <= 100
    _, report = thin_assembly(pieces, faces=10 ** 7)
    assert not report.thinned


@pytest.mark.usefixtures("thinning")
def test_break_labels_follow_the_faces_they_came_from():
    pieces, breaks = _dense()
    (v0, f0), marked = pieces[0], breaks[0] >= 0
    v1, f1 = thin_assembly([pieces[0], pieces[1]], edge=0.05)[0][0]
    moved = transfer_face_labels(v0, f0, marked, v1, f1)
    centroid_z = v1[f1].mean(axis=1)[:, 2]
    top = centroid_z > 0.4 * 30 - 0.05                 # the cap, within the thinning's drift
    # Nearest centroid: every face labelled is on the break, and most of the
    # break's faces are (its source faces here are long fan triangles, whose
    # centroids sit far from the rim).
    assert moved.any() and not moved[~top].any() and moved[top].mean() > 0.5


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def test_entries_are_placed_as_breaking_bad_folders():
    assert place("pig_bone/pig_bone_1") == ("pig_bone", "pig_bone_1")
    assert place("pig/synthetic_fracture/3/mode_1") == ("pig/synthetic_fracture", "3_mode_1")
    assert place("hip/synthetic_fracture/000004740/fractured_12") == \
        ("hip/synthetic_fracture", "000004740_fractured_12")
    assert place("limb1") == ("", "limb1")


def test_garf_files_are_read_with_their_splits_and_pieces_in_numeric_order(tmp_path):
    many, many_breaks = tube_pieces(list(np.linspace(-2.5, 2.5, 11)), rings=60)
    two, two_breaks = tube_pieces([0.4])
    path = write_garf(tmp_path / "bones.hdf5", {"bones/twelve": (many, many_breaks),
                                               "bones/two": (two, two_breaks)})
    assert garf_entries(path) == [("bones/twelve", "val"), ("bones/two", "val")]
    assert garf_entries(path, "train") == []
    assembly = read_garf(path, "bones/twelve", "val")
    assert len(assembly.pieces) == 12
    lowest = [float(v[:, 2].min()) for v, _ in assembly.pieces]
    assert lowest == sorted(lowest)                       # "10" after "9", not after "1"
    assert assembly.category == "bones" and assembly.name == "twelve"
    assert all(b.dtype == bool for b in assembly.breaks)


def test_fantastic_breaks_pairs_and_their_mask_are_read(tmp_path):
    pieces, breaks = tube_pieces([0.4])
    folder = tmp_path / "FB" / "00" / "00002"
    folder.mkdir(parents=True)
    for tag, (v, f) in zip("br", pieces):
        trimesh.Trimesh(v * 30, f, process=False).export(folder / f"model_{tag}_0.ply")
    capped = np.zeros(len(pieces[0][0]), bool)
    capped[np.unique(pieces[0][1][breaks[0] >= 0])] = True
    np.savez(folder / "meta_0.npz", mask=capped)
    assert fantastic_breaks_entries(tmp_path / "FB") == [("00/00002", 0)]
    assembly = read_fantastic_breaks(tmp_path / "FB", "00/00002", 0)
    assert (assembly.category, assembly.name) == ("00", "00002")
    assert len(assembly.pieces) == 2 and assembly.breaks[1] is None
    assert assembly.breaks[0].sum() == (breaks[0] >= 0).sum()


# --------------------------------------------------------------------------
# Preparing and writing
# --------------------------------------------------------------------------

def _assembly(pieces, breaks, name="bones/scan"):
    category, object_name = place(name)
    return Assembly("test", name, category, object_name, "val", list(pieces),
                    [None if b is None else np.asarray(b) >= 0 for b in breaks])


@pytest.mark.usefixtures("thinning")
def test_a_scan_is_cleaned_scaled_and_thinned(real_like):
    stored, marked = real_like
    pieces, row = prepare(_assembly(stored, marked), Preparation(edge=0.05))
    assert row["status"] == "ok" and row["dropped_components"] == 2
    assert row["merged_vertices"] > 0 and row["shared_vertices"] == 0
    assert row["scale"] * row["extent_stored"] == pytest.approx(1.0)    # Breaking Bad's scale
    assert abs(object_extent(pieces) - 1.0) < 1e-3                      # thinning moves little
    assert row["thinned"] and abs(row["edge_after"] / 0.05 - 1) < 0.1
    assert 0 < row["break_share"] < 1 and row["recall"] == row["recall"]   # measured, not NaN


def test_simulated_pieces_keep_the_vertices_they_share():
    pieces, breaks = tube_pieces([-0.5, 0.8], walls=(-1.5,))
    near = median_edge(pieces) / object_extent(pieces)
    prepared, row = prepare(_assembly(pieces, breaks), Preparation(edge=1.5 * near))
    assert row["status"] == "ok" and not row["thinned"]
    assert row["duplicate_faces"] == 2 * 24 and row["shared_vertices"] > 0
    assert row["recall_full"] == row["recall"]


def test_unusable_assemblies_are_named(real_like):
    stored, marked = real_like
    _, row = prepare(_assembly([stored[0], stored[0]], [marked[0], marked[0]]), Preparation())
    assert row["status"] == "skipped: pieces 0 and 1 are the same mesh"
    _, row = prepare(_assembly([stored[0]], [marked[0]]), Preparation())
    assert row["status"].startswith("skipped: 1 piece")


def test_a_small_object_is_scaled_up_to_breaking_bads_size():
    pieces, breaks = tube_pieces([0.4])
    prepared, row = prepare(_assembly([(v * 0.01, f) for v, f in pieces], breaks), Preparation())
    assert row["scale"] > 1 and row["scale"] * row["extent_stored"] == pytest.approx(1.0)
    assert object_extent(prepared) == pytest.approx(1.0)


def test_garf_scale_puts_the_largest_piece_at_one(real_like):
    stored, marked = real_like
    pieces, _ = prepare(_assembly(stored, marked), Preparation(scale="piece"))
    assert abs(max(float(np.ptp(v, axis=0).max()) for v, _ in pieces) - 1.0) < 1e-9
    assert object_extent(pieces) > 1.0


@pytest.mark.usefixtures("thinning")
def test_written_scenes_read_back_through_the_breaking_bad_reader(tmp_path, real_like):
    stored, marked = real_like
    pieces, row = prepare(_assembly(stored, marked, "bones/pig_bone_1"), Preparation(edge=0.05))
    write(pieces, tmp_path / "external" / "bone_real" / row["scene"])
    scene, = find_scenes(tmp_path / "external", ["bone_real"])
    assert (scene.subset, scene.category, scene.name) == ("bone_real", "bones", "pig_bone_1")
    loaded = SceneReader(scene.path).load_mode("fractured_0").fragments
    assert len(loaded) == len(pieces)
    for mesh, (v, f) in zip(loaded, pieces):
        assert len(mesh.faces) == len(f)
        assert np.allclose(np.sort(np.asarray(mesh.vertices), axis=0), np.sort(v, axis=0),
                           rtol=0, atol=1e-8)
    assert not any(tmp_path.rglob("*.part"))


def test_the_reference_density_is_measured_on_breaking_bad_scenes(tmp_path):
    for k, cuts in enumerate(([0.4], [-1.0, 1.0], [0.0])):
        pieces, _ = tube_pieces(cuts, sections=16 + 8 * k)
        write_scene(tmp_path / "data" / "everyday_compressed" / "Tube" / f"t{k}", pieces)
    value, count, (low, high) = reference_edge(tmp_path / "data", ["everyday_compressed"])
    expected = [median_edge(p) / object_extent(p) for p in
                (tube_pieces(c, sections=16 + 8 * k)[0]
                 for k, c in enumerate(([0.4], [-1.0, 1.0], [0.0])))]
    assert count == 3 and value == pytest.approx(float(np.median(expected)), rel=1e-6)
    assert low <= value <= high


# --------------------------------------------------------------------------
# The script, and the model on what it wrote
# --------------------------------------------------------------------------

@pytest.mark.usefixtures("thinning")
def test_the_script_converts_and_the_model_evaluates_the_result(tmp_path, real_like, capsys):
    pytest.importorskip("torch")
    from scripts.convert_external import main
    from reassembly import training
    from reassembly.training import Config, build_model, save_checkpoint

    stored, marked = real_like
    garf = write_garf(tmp_path / "bone_real.hdf5", {
        "pig_bone/pig_bone_1": (stored, marked),
        "pig_bone/pig_bone_15": ([stored[0], stored[0]], [marked[0], marked[0]])})
    fb = tmp_path / "FB_v1" / "00" / "00002"
    fb.mkdir(parents=True)
    for tag, (v, f) in zip("br", tube_pieces([0.4])[0]):
        trimesh.Trimesh(v * 30, f, process=False).export(fb / f"model_{tag}_0.ply")

    with pytest.raises(SystemExit, match="choose the density"):
        main([str(garf), "--out", str(tmp_path / "external")])
    assert main([str(garf), str(tmp_path / "FB_v1"), "--out", str(tmp_path / "external"),
                 "--edge", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "bone_real: 1 of 2 assemblies converted" in out
    assert "1 skipped: pieces 0 and 1 are the same mesh" in out
    assert "FB_v1: 1 of 1 assemblies converted" in out
    with open(tmp_path / "external" / "bone_real.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["status"] for r in rows] == ["ok", "skipped: pieces 0 and 1 are the same mesh"]

    config = Config(root=str(tmp_path / "external"), subsets=["bone_real"],
                    out_dir=str(tmp_path / "out"), channels=16, heads=4, head_dim=4,
                    embedding_dim=8, workers=0, tokens_per_scene=32, batch_size=1,
                    accumulate=1, epochs=1, modes_per_scene=None, schedule=("intra",),
                    check_init=False, steps_per_epoch=0)
    path = Path(config.out_dir) / "last.pt"
    path.parent.mkdir(parents=True)
    save_checkpoint(path, build_model(config), None, config, epoch=0, step=1, history=[],
                    best=1.0)
    summary = training.evaluate(config, checkpoint="last.pt", split="all",
                                predictions=str(tmp_path / "predictions"))
    assert summary["evaluation"]["samples"] == 1 and summary["fragments"] == 2
    assert len(summary["assembly_scenes"]) == 1
    assert list((tmp_path / "predictions").glob("*.zip"))

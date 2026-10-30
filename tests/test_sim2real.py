"""
The sim-to-real changes (``docs/SIM2REAL.md``), in the order that document
gives them:

1. stage two matches by cosine, as the embedding is trained, over candidates
   spread across the whole break; the true break is written beside a
   converted scene, and ``--label_method marked`` -- the oracle -- reads it;
2. isotropic remeshing, the breaks worn, rippled and bent, and one draw per
   scene between remeshed, perturbed and as stored;
3. the fracture head, which picks the cross-fragment tokens on the device and
   decides which vertices stage two matches;
4. the consistency term, between a remeshed or perturbed scene and the same
   scene as stored;
5. the fragment context (``global``).

Every equivariance claim is checked in float64, per fragment -- each fragment
tumbled by its own rotation -- because that is how the data arrives.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

from reassembly.assembly.translation import (farthest_points,  # noqa: E402
                                             mutual_nearest_neighbours, subsample_per_fragment)
from reassembly.data.external import Preparation, prepare, vertex_breaks  # noqa: E402
from reassembly.data.features import collate  # noqa: E402
from reassembly.data.remesh import (Remesh, break_masks, deform_pieces,  # noqa: E402
                                    noise_pieces, remesh_scene, smooth_pieces)
from reassembly.data.scene import BREAK_FILE, SceneReader, write_scene  # noqa: E402
from reassembly.data.transforms import random_rotations  # noqa: E402
from reassembly.mesh.clean import edge_counts  # noqa: E402
from reassembly.mesh.decimate import edge_lengths  # noqa: E402
from reassembly.nn.context import (FALLBACK_TOKENS, _allocate, segment_loss,  # noqa: E402
                                   select_tokens)
from reassembly.nn.losses import consistency_loss  # noqa: E402
from reassembly.nn.model import ReassemblyNet  # noqa: E402
from reassembly.training import (BreakingBadScenes, Config, _collate_samples,  # noqa: E402
                                 _forward, _start_from, build_criterion, build_model,
                                 save_checkpoint)

from test_external import _assembly, real_like, tube_pieces  # noqa: E402,F401
from test_model import DTYPE, _sample  # noqa: E402
from test_training_recovery import _config, _dataset  # noqa: E402


def _isotropic_available() -> bool:
    try:
        from reassembly.data.remesh import _pymeshlab

        _pymeshlab()
        return True
    except ImportError:
        return False


isotropic = pytest.mark.skipif(not _isotropic_available(),
                               reason="needs pymeshlab with its remeshing filters "
                                      "(on Linux, libopengl0)")


@pytest.fixture
def thinning():
    pytest.importorskip("fast_simplification")


def _pieces(meshes):
    return ([np.asarray(m.vertices, dtype=np.float64) for m in meshes],
            [np.asarray(m.faces, dtype=np.int64) for m in meshes])


# --------------------------------------------------------------------------
# 1. Matching by cosine, over the whole break; the true break; the oracle
# --------------------------------------------------------------------------

def test_cosine_matching_compares_directions_not_lengths():
    """
    The contrastive term compares embeddings by cosine and nothing trains
    their length. Matched by Euclidean distance (the default, v7's), two
    embeddings pointing the same way but of different lengths can fail to
    match; with ``--match_metric cosine`` they cannot. With the lengths
    scrambled, cosine still finds every partner and Euclidean distance does
    not.
    """
    rng = np.random.default_rng(0)
    base = torch.as_tensor(rng.normal(size=(12, 8)))
    embeddings = torch.cat([base, base])
    embeddings[12:] *= torch.as_tensor(rng.uniform(0.05, 20.0, size=(12, 1)))
    owner = torch.tensor([0] * 12 + [1] * 12)

    def found(source, target):
        return sum((t - s) == 12 for s, t in zip(source.tolist(), target.tolist()))

    assert found(*mutual_nearest_neighbours(embeddings, owner, cosine=True)) == 12
    assert found(*mutual_nearest_neighbours(embeddings, owner)) < 12
    with pytest.raises(ValueError, match="match_metric"):
        Config(match_metric="manhattan")


def test_the_candidates_cover_the_break_and_do_not_depend_on_its_pose():
    """
    A dense scan's vertices come in the scanner's order, and a stride through
    that order keeps some regions and skips others. Farthest-point sampling
    spreads the subset over the space the points cover -- and picks the same
    vertices whichever way the fragment is turned.
    """
    rng = np.random.default_rng(1)
    dense = np.c_[rng.uniform(0.0, 0.1, 900), rng.uniform(0, 1, 900), np.zeros(900)]
    sparse = np.c_[rng.uniform(0.1, 1.0, 100), rng.uniform(0, 1, 100), np.zeros(100)]
    points = torch.as_tensor(np.vstack([dense, sparse]))
    chosen = farthest_points(points, 60)
    assert chosen.unique().numel() == 60
    # The spread: most of the subset outside the dense tenth, where a stride
    # by index would take 54 of the 60 inside it.
    assert float((points[chosen, 0] > 0.1).double().mean()) > 0.6
    turn = torch.as_tensor(random_rotations(1, np.random.default_rng(2))[0])
    assert torch.equal(farthest_points(points @ turn.T, 60), chosen)

    fragment = torch.tensor([0] * 1000 + [1] * 30)
    stacked = torch.cat([points, torch.zeros(30, 3, dtype=points.dtype)])
    keep = subsample_per_fragment(fragment, 50, points=stacked)
    assert (fragment[keep] == 0).sum() == 50 and (fragment[keep] == 1).sum() == 30
    assert torch.equal(keep[:50], torch.sort(keep[:50]).values)


def _two_spheres():
    a = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    b = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    b.apply_translation([3.0, 0.0, 0.0])
    return [(np.asarray(a.vertices), np.asarray(a.faces)),
            (np.asarray(b.vertices), np.asarray(b.faces))]


def test_a_converted_scene_keeps_its_true_break(tmp_path):
    """Written beside the scene, read back per fragment as loaded -- the
    loader cleans each piece and may reorder its vertices, so the marks are
    found again by position."""
    pieces = _two_spheres()
    marks = [v[:, 2] > 0.3 for v, _ in pieces]
    write_scene(tmp_path / "scene", pieces, breaks=marks)
    reader = SceneReader(tmp_path / "scene")
    result = reader.load_mode("fractured_0")
    read = reader.break_masks(result)
    assert len(read) == 2
    for fragment, mask in zip(result.fragments, read):
        assert np.array_equal(mask, np.asarray(fragment.vertices)[:, 2] > 0.3)
    write_scene(tmp_path / "plain", pieces)
    plain = SceneReader(tmp_path / "plain")
    assert not (tmp_path / "plain" / "fractured_0" / BREAK_FILE).exists()
    assert plain.break_masks(plain.load_mode("fractured_0")) is None
    with pytest.raises(ValueError, match="one bool per vertex"):
        write_scene(tmp_path / "bad", pieces, breaks=[marks[0], marks[1][:-1]])


def test_a_piece_with_no_marks_takes_the_break_beside_the_marked_one():
    """Fantastic Breaks marks the broken object only; its restoration takes
    the vertices within BREAK_REACH of the marked break."""
    left = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).subdivide().subdivide()
    left.apply_translation([-0.5, 0.0, 0.0])
    right = left.copy()
    right.apply_translation([1.01, 0.0, 0.0])                      # 0.01 apart
    v0, f0 = np.asarray(left.vertices), np.asarray(left.faces)
    on_face = np.all(np.isclose(v0[f0][:, :, 0], 0.0), axis=1)     # the facing side
    masks = vertex_breaks([(v0, f0), (np.asarray(right.vertices), np.asarray(right.faces))],
                          [on_face, None])
    assert np.array_equal(masks[0], np.isclose(v0[:, 0], 0.0))
    assert np.array_equal(masks[1], np.asarray(right.vertices)[:, 0] < 0.02)
    assert vertex_breaks([(v0, f0)], [None]) is None


@pytest.mark.usefixtures("thinning")
def test_the_converter_returns_the_marked_break_with_the_pieces(real_like):  # noqa: F811
    stored, marked = real_like
    pieces, row, breaks = prepare(_assembly(stored, marked), Preparation(edge=0.05),
                                  return_breaks=True)
    assert row["status"] == "ok" and row["marked_pieces"] == 2
    assert [len(b) for b in breaks] == [len(v) for v, _ in pieces]
    # The tube's cut is at z = 0.4 of [-3, 3], scaled to a largest side of 1.
    for (vertices, _), mask in zip(pieces, breaks):
        assert mask.any() and not mask.all()
        z = vertices[mask][:, 2]
        assert np.ptp(z) < 0.02
    again, _ = prepare(_assembly(stored, marked), Preparation(edge=0.05))
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(again, pieces))


def test_a_dataset_that_marks_nothing_has_its_break_estimated(fractured_solid):
    """
    Fractura's real bones mark no break (their shared_faces mark the debris the
    cleaning drops, or nothing). The estimate -- the nearest face of another
    piece within reach, the face normals opposed -- is checked where the truth
    is known: the fixture's coincident interface, and the tube's caps opened a
    little apart. By vertex normals, which a rim vertex averages over the break
    and the intact side, the fixture's recall was 0.84; by faces, all of it.
    """
    from reassembly.data.external import Assembly, estimated_breaks
    from reassembly.mesh.fracture import fracture_vertex_masks

    vertices, faces = _pieces(fractured_solid)
    truth = fracture_vertex_masks(vertices)
    for (v, f), face_mask, t in zip(zip(vertices, faces), estimated_breaks(list(zip(vertices, faces)),
                                                                             reach=0.05), truth):
        estimated = np.zeros(len(v), bool)
        estimated[np.unique(f[face_mask])] = True
        precision = (estimated & t).sum() / max(estimated.sum(), 1)
        assert precision > 0.95 and (estimated & t).sum() > 0.95 * t.sum()
    pieces, breaks = tube_pieces([0.4])
    pieces[1] = (pieces[1][0] + [0.0, 0.0, 0.02], pieces[1][1])       # opened by 0.02 of 6
    assembly = Assembly("test", "tube", "", "tube", "", list(pieces), [None, None])
    _, row, marks = prepare(assembly, Preparation(), return_breaks=True)
    assert row["break_source"] == "estimated"
    for (v, f), b, mask in zip(pieces, breaks, marks):
        cap = np.zeros(len(v), bool)
        cap[np.unique(f[b >= 0])] = True
        assert (mask & cap).sum() / max(mask.sum(), 1) > 0.9 and (mask & cap).sum() > 0.95 * cap.sum()
    marked = Assembly("test", "tube", "", "tube", "", list(pieces), [b >= 0 for b in breaks])
    assert prepare(marked, Preparation())[1]["break_source"] == "marked"


@isotropic
def test_a_dense_scan_is_thinned_keeping_its_topology_before_it_is_remeshed():
    from reassembly.data.remesh import thin_keeping_topology

    from reassembly.mesh.decimate import surface_area

    dense = trimesh.creation.icosphere(subdivisions=6)              # 81,920 faces
    v0, f0 = np.asarray(dense.vertices), np.asarray(dense.faces)
    v, f = thin_keeping_topology(v0, f0, 0.05)
    even = surface_area(v0, f0) / (np.sqrt(3.0) / 4.0 * 0.05 ** 2)  # an even mesh's faces
    assert abs(len(f) / even - 1) < 0.05 and edge_counts(f) == (0, 0)
    assert thin_keeping_topology(v, f, 0.01)[1] is f                # already coarser: kept


def _marked_dataset(tmp_path):
    """The fixture's scenes, each mode carrying a true break that is NOT the
    coincident vertices -- so the test can tell the two apart."""
    root = _dataset(tmp_path / "data")
    for mesh_file in root.rglob("compressed_mesh.obj"):
        vertices = np.asarray(trimesh.load(mesh_file, process=False).vertices)
        for mode in mesh_file.parent.glob("fractured_*"):
            np.save(mode / BREAK_FILE, vertices[:, 2] > 0.5)
    return root


def test_marked_labels_are_the_true_break(tmp_path):
    """
    ``--label_method marked`` reads the break a converted scene stores, and on
    Breaking Bad -- no file -- the coincident vertices, carried over to the
    remeshed pieces (which ``coincidence`` cannot label).
    """
    root = _dataset(tmp_path / "data")
    marked = BreakingBadScenes(_config(root, label_method="marked"), "train")
    coincident = BreakingBadScenes(_config(root, label_method="coincidence"), "train")
    a, b = marked.build(0), coincident.build(0)
    assert all(np.array_equal(x.fracture, y.fracture) for x, y in zip(a.fragments, b.fragments))
    assert np.array_equal(a.breaks, np.concatenate([f.fracture for f in a.fragments]))
    assert b.breaks is None                         # nothing reads the truth there

    stored = _marked_dataset(tmp_path / "stored")
    scenes = BreakingBadScenes(_config(stored, label_method="marked"), "train")
    sample = scenes.build(0)
    for fragment, mesh in zip(sample.fragments, scenes.meshes(0)):
        assert np.array_equal(fragment.fracture, np.asarray(mesh.vertices)[:, 2] > 0.5)


@pytest.mark.usefixtures("thinning")
def test_marked_labels_follow_the_pieces_when_they_are_remeshed(tmp_path):
    root = _dataset(tmp_path / "data")
    scenes = BreakingBadScenes(_config(root, label_method="marked", remesh=(0.5, 0.9)),
                               "train")
    sample = scenes.build(0)
    stored, _ = _pieces(scenes.meshes(0))
    expected = break_masks(stored, [p.vertices for p in scenes.pieces(0)])
    assert all(np.array_equal(f.fracture, m) for f, m in zip(sample.fragments, expected))
    assert np.array_equal(sample.breaks, np.concatenate(expected))


# --------------------------------------------------------------------------
# 2. Isotropic remeshing, the breaks made to look scanned, one draw per scene
# --------------------------------------------------------------------------

def _turned(before, after, faces) -> int:
    normal = lambda v: np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])  # noqa: E731
    return int((np.einsum("ij,ij->i", normal(before), normal(after)) <= 0).sum())


def test_the_breaks_are_worn_rippled_and_bent_without_a_face_turning(fractured_solid):
    vertices, faces = _pieces(fractured_solid)
    breaks = break_masks(vertices, vertices)
    rng = np.random.default_rng(0)
    worn = smooth_pieces(vertices, faces, breaks, rng, 6)
    for v0, v1, f, mask in zip(vertices, worn, faces, breaks):
        moved = np.linalg.norm(v1 - v0, axis=1) > 0
        assert moved.any() and _turned(v0, v1, f) == 0
        # Only the break and the ring beside it move.
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        beside = np.zeros(len(v0), bool)
        beside[edges[mask[edges[:, 0]], 1]] = True
        assert not moved[~(mask | beside)].any()
    rippled = noise_pieces(vertices, faces, np.random.default_rng(1), 0.02)
    for v0, v1, f in zip(vertices, rippled, faces):
        assert 0 < np.abs(v1 - v0).max() < 0.2 and _turned(v0, v1, f) == 0
    bent = deform_pieces(vertices, np.random.default_rng(2), 0.02)
    for v0, v1, f in zip(vertices, bent, faces):
        assert 0 < np.abs(v1 - v0).max() < 0.2 and _turned(v0, v1, f) == 0
    # Off, nothing moves -- and no number is drawn.
    rng = np.random.default_rng(3)
    assert all(np.array_equal(a, b) for a, b in zip(smooth_pieces(vertices, faces, breaks,
                                                                 rng, 0), vertices))
    assert all(np.array_equal(a, b) for a, b in zip(noise_pieces(vertices, faces, rng, 0.0),
                                                    vertices))
    assert rng.uniform() == np.random.default_rng(3).uniform()


def test_the_labels_are_read_before_the_ripple_and_the_positives_before_it_all(
        fractured_solid):
    vertices, faces = _pieces(fractured_solid)
    spec = Remesh(1.0, 1.0, jitter=1e-3, gap=0.2, smooth=3, noise=0.01, deform=0.01)
    closed, surface, moved, kept = remesh_scene(vertices, faces, np.random.default_rng(0), spec)
    assert all(np.array_equal(a, b) for a, b in zip(kept, faces))   # LOW = HIGH = 1: no collapse
    assert all(np.array_equal(a, b) for a, b in zip(closed, vertices))
    assert all(np.abs(a - b).max() > 0 for a, b in zip(surface, closed))
    assert all(np.abs(a - b).max() > 0 for a, b in zip(moved, surface))
    same = remesh_scene(vertices, faces, np.random.default_rng(0), spec)
    assert all(np.array_equal(a, b) for a, b in zip(same[2], moved))
    stored = remesh_scene(vertices, faces, np.random.default_rng(0), spec, remesh=False,
                          perturb=False)
    assert all(np.array_equal(a, b) for stage in stored[:3] for a, b in zip(stage, vertices))


@isotropic
def test_isotropic_remeshing_evens_the_edges_and_takes_the_twins_away(fractured_solid):
    from reassembly.data.remesh import isotropic_pieces
    from reassembly.mesh.fracture import fracture_vertex_masks

    vertices, faces = _pieces(fractured_solid)
    remeshed, kept = isotropic_pieces(vertices, faces, np.random.default_rng(0), 0.12, 0.12)
    for v, f in zip(remeshed, kept):
        assert edge_counts(f) == (0, 0)                     # still closed and manifold
        assert abs(np.median(edge_lengths(v, f)) / 0.12 - 1) < 0.15
    # The two sides of the break are tessellated each on its own: almost no
    # vertex is left where the other piece has one (1,152 before).
    shared = sum(m.sum() for m in fracture_vertex_masks(remeshed))
    assert shared < 0.05 * sum(m.sum() for m in fracture_vertex_masks(vertices))
    # Long thin triangles, as a simulation leaves them, come out even.
    bar = trimesh.creation.box(extents=(2.0, 0.2, 0.2))
    v0, f0 = np.asarray(bar.vertices), np.asarray(bar.faces)
    for _ in range(3):
        v0, f0 = trimesh.remesh.subdivide(v0, f0)
    ratio = lambda e: np.percentile(e, 90) / np.percentile(e, 10)  # noqa: E731
    (v1,), (f1,) = isotropic_pieces([v0], [f0], np.random.default_rng(0), 0.05, 0.05)
    assert ratio(edge_lengths(v1, f1)) < 0.5 * ratio(edge_lengths(v0, f0))
    # The two sides drew their own edge length, within the range.
    _, other = isotropic_pieces(vertices, faces, np.random.default_rng(0), 0.1, 0.2)
    assert len(other[0]) != len(other[1])


def _finned_sphere():
    """A sphere with 300 fins: triangles hung on its edges, each edge then
    shared by three faces -- a stand-in for a boolean cut's mesh, such as
    Fantastic Breaks' restorations, which no remesher can make even."""
    sphere = trimesh.creation.icosphere(subdivisions=3)
    v, f = np.asarray(sphere.vertices), np.asarray(sphere.faces)
    tips, fins = [], []
    for k in range(300):
        a, b = f[k % len(f)][:2]
        tips.append((v[a] + v[b]) / 2 + np.random.default_rng(k).normal(0, 0.3, 3))
        fins.append([a, b, len(v) + k])
    return np.vstack([v, tips]), np.vstack([f, fins])


@isotropic
def test_a_piece_the_remesher_cannot_work_on_is_refused_and_handled(tmp_path):
    from reassembly.data.external import Assembly
    from reassembly.data.remesh import remesh_isotropic, try_isotropic

    vertices, faces = _finned_sphere()
    assert edge_counts(faces)[1] == 300
    # Training: collapsed instead, to the faces an even mesh of that edge has --
    # or, the piece already coarser than that, kept as it is.
    assert try_isotropic(vertices, faces, 0.3) is None
    v, f = remesh_isotropic(vertices, faces, 0.3)
    assert len(f) < len(faces) / 2
    assert try_isotropic(vertices, faces, 0.08) is None
    v, f = remesh_isotropic(vertices, faces, 0.08)
    assert np.array_equal(f, faces) and np.array_equal(v, vertices)
    # A manifold piece passes the gate.
    sphere = trimesh.creation.icosphere(subdivisions=4)
    assert try_isotropic(np.asarray(sphere.vertices), np.asarray(sphere.faces), 0.08) is not None
    # The converter collapses it to the density instead, its topology kept,
    # and counts it -- or, already coarser than that, keeps it as it came.
    good = trimesh.creation.icosphere(subdivisions=4)
    good.apply_translation([2.2, 0.0, 0.0])
    pieces = [(np.asarray(good.vertices), np.asarray(good.faces)), (vertices, faces)]
    marks = [np.zeros(len(good.faces), bool), np.zeros(len(faces), bool)]
    marks[0][:40] = True
    assembly = Assembly("test", "fins", "", "fins", "", pieces, marks)
    out, row, breaks = prepare(assembly, Preparation(edge=0.018, isotropic=True),
                               return_breaks=True)
    assert row["status"] == "ok" and row["isotropic_failed"] == 1
    assert len(out[1][1]) == len(faces) and edge_counts(out[1][1])[1] > 0
    assert edge_counts(out[0][1]) == (0, 0) and breaks[0].any()
    out, row = prepare(assembly, Preparation(edge=0.08, isotropic=True))
    assert row["status"] == "ok" and row["isotropic_failed"] == 1
    assert len(out[1][1]) < len(faces) / 2 and edge_counts(out[1][1])[1] > 0


def test_one_draw_per_scene_decides_remeshed_perturbed_or_stored(tmp_path):
    root = _dataset(tmp_path / "data")
    config = _config(root, remesh=(1.0, 1.0), remesh_gap=0.3, contact_radius=0.35,
                     remesh_probability=0.5, augment_probability=0.5)
    decisions = []
    for epoch in range(4):
        scenes = BreakingBadScenes(config, "train", epoch_seed=epoch)
        for i in range(len(scenes)):
            remeshed, perturbed = scenes._decisions(scenes.key(i))
            decisions.append((remeshed, perturbed))
            if not remeshed:
                stored = scenes.meshes(i)
                assert all(np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices))
                           for a, b in zip(scenes.pieces(i), stored))
    assert all(r == p for r, p in decisions)        # one draw, both shares 0.5
    assert {r for r, _ in decisions} == {True, False}
    # Every scene remeshed, none perturbed: the pieces stay where the remesher put them.
    calm = BreakingBadScenes(_config(root, remesh=(1.0, 1.0), remesh_gap=0.3,
                                     contact_radius=0.35, augment_probability=0.0), "train")
    assert all(calm._decisions(calm.key(i)) == (True, False) for i in range(len(calm)))
    closed, _, seen = calm._remeshed(calm.key(0), *_pieces(calm.meshes(0)))
    assert all(np.array_equal(a.vertices, b.vertices) for a, b in zip(closed, seen))


def test_the_training_loader_deals_every_step_its_share_of_stored_scenes(tmp_path):
    """
    With both shares at 0.625, every optimizer step -- ``batch_size *
    accumulate`` scenes on one GPU -- holds exactly 5 of 8 remeshed and
    perturbed and 3 as stored: the sampler deals each scene its number, and the
    scene honours it. Validation, and a run that perturbs every scene, are not
    dealt anything.
    """
    from reassembly.data.sampling import DealtShares
    from reassembly.training import _contact_banner, _loader

    root = _dataset(tmp_path / "data")
    config = _config(root, remesh=(1.0, 1.0), remesh_gap=0.3, contact_radius=0.35,
                     remesh_probability=0.625, augment_probability=0.625,
                     batch_size=2, accumulate=4, steps_per_epoch=12)
    scenes = BreakingBadScenes(config, "train", epoch_seed=1)
    loader = _loader(scenes, config, True, 0, 1, epoch=1)
    assert isinstance(loader.sampler, DealtShares) and loader.sampler.block == 8
    pairs = list(loader.sampler)
    assert len(pairs) == 12 * 2
    for start in range(0, len(pairs), 8):
        decided = [scenes._decisions(scenes.key(i), share) for i, share in pairs[start:start + 8]]
        assert decided.count((True, True)) == 5 and decided.count((False, False)) == 3
    index = pairs[0][0]
    stored = scenes.meshes(index)
    for share, changed in ((0.1, True), (0.9, False)):
        _closed, _surface, seen = scenes._remeshed(scenes.key(index), *_pieces(stored), share)
        same = all(np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices))
                   for a, b in zip(seen, stored))
        assert same is not changed
    built, again = scenes[(index, 0.9)], scenes.build(index, share=0.9)
    assert all(np.array_equal(a.target_vertices, b.target_vertices)
               for a, b in zip(built.fragments, again.fragments))
    assert "of each optimizer step's 8 scenes per GPU, 5 remeshed and perturbed, 3 as stored" in \
        "\n".join(_contact_banner(config))
    assert not isinstance(_loader(BreakingBadScenes(config, "val"), config, False, 0, 1,
                                  epoch=1).sampler, DealtShares)
    every = _config(root, remesh=(1.0, 1.0), remesh_gap=0.3, contact_radius=0.35,
                    batch_size=2, accumulate=4, steps_per_epoch=12)
    assert not isinstance(_loader(BreakingBadScenes(every, "train"), every, True, 0, 1,
                                  epoch=1).sampler, DealtShares)


def test_val_remesh_validates_on_every_scene_remeshed_the_same_way_each_epoch(tmp_path):
    """
    ``--val_remesh``: every validation scene remeshed by the training
    settings, whatever ``--remesh_probability``, and ``--augment_probability``
    of them perturbed, by one draw per scene that every epoch repeats -- and
    the positives still found before anything moved. Without it validation is
    as stored, and an evaluation never takes it from a checkpoint.
    """
    from reassembly.training import _TRAINING_ONLY, _contact_banner

    root = _dataset(tmp_path / "data")
    config = _config(root, remesh=(1.0, 1.0), remesh_gap=0.3, contact_radius=0.35,
                     remesh_probability=0.625, augment_probability=1.0, val_remesh=True)
    val = BreakingBadScenes(config, "val")
    assert val.remesh is not None and val.remesh.remesh_probability == 1.0
    stored = BreakingBadScenes(dataclasses.replace(config, val_remesh=False), "val")
    assert stored.remesh is None
    shared = BreakingBadScenes(dataclasses.replace(config, augment_probability=0.5), "val")
    decided = [shared._decisions(shared.key(i)) for i in range(len(shared))]
    assert all(remeshed for remeshed, _ in decided)
    assert decided == [shared._decisions(shared.key(i)) for i in range(len(shared))]
    for i in range(len(val)):
        assert val._decisions(val.key(i)) == (True, True)
        once, again = val.build(i), BreakingBadScenes(config, "val").build(i)
        assert all(np.array_equal(a.target_vertices, b.target_vertices)
                   for a, b in zip(once.fragments, again.fragments))
        plain = stored.build(i)
        assert not all(np.array_equal(a.target_vertices, b.target_vertices)
                       for a, b in zip(once.fragments, plain.fragments))
        assert once.contact is not None and once.contact.any()
        assert once.contact_points is not None and once.view is None
    assert "validation remeshed too, every scene, 100.0% of them perturbed" in \
        "\n".join(_contact_banner(config))
    assert "val_remesh" in _TRAINING_ONLY
    with pytest.raises(ValueError, match="val_remesh"):
        _config(root, contact_radius=0.35, val_remesh=True)


def test_val_augment_probability_sets_the_share_of_validation_scenes_perturbed(tmp_path, capsys):
    """
    ``--val_augment_probability``: with ``--val_remesh``, the share of the
    validation scenes perturbed in place of the training share -- 1 perturbs
    every one while training keeps its mix -- by the same one draw per scene.
    Left out, validation follows ``--augment_probability`` as before. Refused
    without ``--val_remesh`` or outside [0, 1]. A resume onto another share is
    announced and resets the best-so-far; the same share spelled out is no
    change, and an evaluation never takes it from a checkpoint.
    """
    from reassembly.training import (_DATA, _TRAINING_ONLY, _check_resume_compatible,
                                     _contact_banner, _fragment_limit_changed,
                                     validation_perturbed)
    from scripts.config_flags import config_flags
    from scripts.train import build_parser, config_from_args

    root = _dataset(tmp_path / "data")
    mix = dict(remesh=(1.0, 1.0), remesh_gap=0.3, contact_radius=0.35,
               remesh_probability=0.625, augment_probability=0.625, val_remesh=True)
    follows = _config(root, **mix)
    every = _config(root, **mix, val_augment_probability=1.0)
    assert (validation_perturbed(follows), validation_perturbed(every)) == (0.625, 1.0)
    assert validation_perturbed(dataclasses.replace(every, val_remesh=False,
                                                    val_augment_probability=None)) is None

    val = BreakingBadScenes(every, "val")
    assert (val.remesh.remesh_probability, val.remesh.augment_probability) == (1.0, 1.0)
    assert all(val._decisions(val.key(i)) == (True, True) for i in range(len(val)))
    train = BreakingBadScenes(every, "train")
    assert (train.remesh.remesh_probability, train.remesh.augment_probability) == (0.625, 0.625)
    assert BreakingBadScenes(follows, "val").remesh.augment_probability == 0.625
    unperturbed = BreakingBadScenes(dataclasses.replace(every, val_augment_probability=0.0), "val")
    assert all(unperturbed._decisions(unperturbed.key(i)) == (True, False)
               for i in range(len(unperturbed)))
    for i in range(len(val)):
        once, again = val.build(i), BreakingBadScenes(every, "val").build(i)
        assert all(np.array_equal(a.target_vertices, b.target_vertices)
                   for a, b in zip(once.fragments, again.fragments))
        plain = unperturbed.build(i)
        assert not all(np.array_equal(a.target_vertices, b.target_vertices)
                       for a, b in zip(once.fragments, plain.fragments))
    assert ("every scene, 100.0% of them perturbed, the same draw every epoch "
            "(--val_remesh, --val_augment_probability)") in "\n".join(_contact_banner(every))
    assert "62.5% of them perturbed, the same draw every epoch (--val_remesh)" in \
        "\n".join(_contact_banner(follows))

    with pytest.raises(ValueError, match="give --val_remesh True"):
        _config(root, **{**mix, "val_remesh": False}, val_augment_probability=1.0)
    with pytest.raises(ValueError, match="a share of the scenes"):
        _config(root, **mix, val_augment_probability=1.5)

    parse = lambda argv: config_from_args(build_parser().parse_args(argv))  # noqa: E731
    base = ["--remesh_edge", "0.012", "0.018", "--val_remesh", "True"]
    assert parse(base).val_augment_probability is None
    assert parse(base + ["--val_augment_probability", "1"]).val_augment_probability == 1.0
    written = config_flags(every, fields=("val_augment_probability",))
    assert written == ["--val_augment_probability", "1.0"]
    assert parse(base + written).val_augment_probability == 1.0

    assert "val_augment_probability" in _DATA and "val_augment_probability" in _TRAINING_ONLY
    saved = dataclasses.asdict(follows)
    assert _fragment_limit_changed(saved, every)
    assert not _fragment_limit_changed(saved, dataclasses.replace(follows,
                                                                  val_augment_probability=0.625))
    before = {k: v for k, v in saved.items() if k != "val_augment_probability"}
    assert not _fragment_limit_changed(before, follows)        # a checkpoint from before it
    _check_resume_compatible(every, before, announce=True)
    assert "val_augment_probability: None -> 1.0" in capsys.readouterr().out


def test_settings_that_cannot_work_are_refused(tmp_path):
    root = _dataset(tmp_path / "data")
    with pytest.raises(ValueError, match="after segment"):
        _config(root, schedule=("intra", "cross", "segment", "intra"))
    with pytest.raises(ValueError, match="one segment layer"):
        _config(root, schedule=("intra", "segment", "segment", "cross"))
    with pytest.raises(ValueError, match="unknown layer kinds"):
        _config(root, schedule=("intra", "pool"))
    with pytest.raises(ValueError, match="give one of them"):
        _config(root, remesh=(0.5, 0.9), remesh_edge=(0.01, 0.02))
    with pytest.raises(ValueError, match="give --remesh"):
        _config(root, remesh_smooth=3)
    with pytest.raises(ValueError, match="needs scenes that are not"):
        _config(root, w_consistency=1.0)
    with pytest.raises(ValueError, match="share of the scenes"):
        _config(root, remesh=(0.5, 0.9), augment_probability=1.5)
    with pytest.raises(ValueError, match="remesh_edge is two edge lengths"):
        _config(root, remesh_edge=(0.02, 0.01))
    _config(root, label_method="marked", remesh_edge=(0.01, 0.02), remesh_smooth=2,
            schedule=("intra", "segment", "global", "cross", "intra"))


# --------------------------------------------------------------------------
# 3. The fracture head and the tokens it picks; 5. the fragment context
# --------------------------------------------------------------------------

SCHEDULE = ("intra", "intra", "segment", "global", "cross", "intra", "cross", "intra", "global")


def _net(schedule=SCHEDULE):
    torch.manual_seed(0)
    return ReassemblyNet(channels=16, heads=4, schedule=schedule).to(DTYPE)


def _run(net, batch, mask=None, budget=40):
    return net(batch.node_features, batch.edge_index, batch.edge_attr, batch.vertex_fragment,
               batch.num_fragments, log_scale=batch.log_scale, token_index=batch.token_index,
               token_query=batch.token_query, token_key=batch.token_key,
               fragment_scene=batch.fragment_scene, num_scenes=batch.num_scenes,
               token_mask=mask, tokens_per_scene=budget)


def test_the_fracture_head_and_the_context_keep_the_network_equivariant_per_fragment():
    """
    Each fragment tumbled by its own rotation: its features turn with it, its
    embedding and its fracture logits do not move, and the network picks the
    same tokens -- the choice reads the logits and distances within one
    fragment, all invariant. (The spheres here are symmetric, so many
    candidates tie; the ties are broken by vertex order, not by round-off.)
    """
    net = _net()
    rotations = random_rotations(3, np.random.default_rng(9))
    identity = np.repeat(np.eye(3)[None], 3, axis=0)
    still = collate([_sample(3, rotations=identity)], dtype=DTYPE)
    turned = collate([_sample(3, rotations=rotations)], dtype=DTYPE)
    a, b = _run(net, still), _run(net, turned)
    Q = torch.as_tensor(rotations, dtype=DTYPE)
    expected = torch.einsum("nij,nkj->nki", Q[still.vertex_fragment], a.vertex_features)
    assert torch.allclose(b.vertex_features, expected, atol=1e-9)
    assert torch.allclose(b.vertex_embedding, a.vertex_embedding, atol=1e-9)
    assert torch.allclose(b.fracture_logit, a.fracture_logit, atol=1e-9)
    assert torch.equal(a.token_index, b.token_index) and a.token_index.numel() == 40
    # Every parameter learns: the head through its own term, the rest through both.
    (b.vertex_embedding.square().sum() + b.fracture_logit.square().sum()).backward()
    assert all(p.grad is not None and p.grad.abs().max() > 0 for p in net.parameters())


def test_one_fragment_turned_leaves_the_others_alone():
    net = _net()
    base = random_rotations(3, np.random.default_rng(1))
    moved = base.copy()
    moved[1] = random_rotations(1, np.random.default_rng(2))[0] @ base[1]
    a = _run(net, collate([_sample(4, rotations=base)], dtype=DTYPE))
    batch = collate([_sample(4, rotations=moved)], dtype=DTYPE)
    b = _run(net, batch)
    other = batch.vertex_fragment != 1
    assert torch.allclose(a.vertex_features[other], b.vertex_features[other], atol=1e-9)
    assert torch.allclose(a.vertex_embedding, b.vertex_embedding, atol=1e-9)


def test_tokens_stay_within_the_budget_and_the_mask():
    batch = collate([_sample(1), _sample(2)], dtype=DTYPE)
    mask = batch.fracture
    tokens = select_tokens(mask, batch.node_features[:, 0, :], batch.vertex_fragment,
                           batch.fragment_scene, batch.num_fragments, batch.num_scenes, 30)
    assert bool(mask[tokens].all()) and torch.equal(tokens, torch.sort(tokens).values)
    scene = batch.fragment_scene[batch.vertex_fragment[tokens]]
    assert torch.bincount(scene, minlength=2).tolist() == [30, 30]
    # Split by how many eligible vertices each fragment has (24 each here).
    per = torch.bincount(batch.vertex_fragment[tokens], minlength=6)
    assert per.max() - per.min() <= 1
    # A budget above the eligible count takes them all.
    every = select_tokens(mask, batch.node_features[:, 0, :], batch.vertex_fragment,
                          batch.fragment_scene, batch.num_fragments, batch.num_scenes, 500)
    assert torch.equal(every, torch.nonzero(mask).flatten())
    # A fragment the head finds nothing on still gets its likeliest few; the
    # truth (no score) is taken as it is.
    empty = mask & (batch.vertex_fragment != 0)
    score = -torch.arange(len(mask), dtype=DTYPE)
    picked = select_tokens(empty, batch.node_features[:, 0, :], batch.vertex_fragment,
                           batch.fragment_scene, batch.num_fragments, batch.num_scenes, 30,
                           score=score)
    assert 0 < int((batch.vertex_fragment[picked] == 0).sum()) <= FALLBACK_TOKENS
    truth = select_tokens(empty, batch.node_features[:, 0, :], batch.vertex_fragment,
                          batch.fragment_scene, batch.num_fragments, batch.num_scenes, 30)
    assert int((batch.vertex_fragment[truth] == 0).sum()) == 0


def test_the_budget_is_split_exactly():
    counts = torch.tensor([500, 3, 0, 1, 120])
    quota = _allocate(counts, 64)
    assert int(quota.sum()) == 64 and bool((quota <= counts).all())
    assert bool((quota[counts > 0] >= 1).all()) and int(quota[2]) == 0
    assert torch.equal(_allocate(torch.tensor([3, 4]), 64), torch.tensor([3, 4]))


def test_the_fracture_term_reads_its_reference_values():
    """
    Balanced weights average to 1 over a fragment, so an untrained head (p =
    0.5) reads ln 2, and a quarter of that with the focal exponent 2. A head
    that calls everything intact -- the cheap answer when breaks are a tenth
    of the vertices -- scores far worse than one that is right.
    """
    fragment = torch.tensor([0] * 50 + [1] * 50)
    target = torch.zeros(100)
    target[torch.tensor([0, 1, 2, 3, 4, 50, 51, 52, 53, 54])] = 1.0
    zero = torch.zeros(100, dtype=torch.float64)
    assert float(segment_loss(zero, target, fragment, 2, gamma=0.0)) == pytest.approx(np.log(2))
    assert float(segment_loss(zero, target, fragment, 2, gamma=2.0)) == pytest.approx(
        np.log(2) / 4)
    right = (target.double() * 2 - 1) * 8.0
    lazy = torch.full((100,), -8.0, dtype=torch.float64)
    assert float(segment_loss(right, target, fragment, 2, gamma=0.0)) < 1e-3
    assert float(segment_loss(lazy, target, fragment, 2, gamma=0.0)) > 3.0
    # Unknown vertices are left out, and with none known the term is 0.
    keep = fragment == 0
    assert float(segment_loss(zero, target, fragment, 2, gamma=0.0, keep=keep)) == pytest.approx(
        np.log(2))
    assert float(segment_loss(zero, target, fragment, 2, keep=torch.zeros(100, dtype=bool))) == 0


def test_the_consistency_term_reads_zero_for_agreement_and_one_for_unrelated():
    rng = np.random.default_rng(0)
    view = torch.tensor(rng.normal(size=(20, 8)), requires_grad=True)
    index = torch.arange(20)
    same = view.detach().clone().requires_grad_(True)
    assert float(consistency_loss(same, view, index).detach()) == pytest.approx(0.0, abs=1e-12)
    away = -view.detach()
    assert float(consistency_loss(away, view, index)) == pytest.approx(2.0)
    unrelated = torch.as_tensor(rng.normal(size=(4000, 8)))
    targets = torch.as_tensor(rng.normal(size=(4000, 8)))
    assert float(consistency_loss(unrelated, targets, torch.arange(4000),
                                  max_anchors=4000)) == pytest.approx(1.0, abs=0.05)
    # -1 is no partner; the view is the target and takes no gradient.
    partial = torch.where(index < 10, index, torch.full_like(index, -1))
    loss = consistency_loss(same + 1.0, view, partial)
    loss.backward()
    assert view.grad is None and bool((same.grad[10:] == 0).all())
    assert float(consistency_loss(same, view, torch.full_like(index, -1)).detach()) == 0.0


def _segmenting(root, **kwargs):
    return _config(root, schedule=("intra", "segment", "global", "cross", "intra"),
                   contact_radius=0.35, **kwargs)


def test_the_forward_trains_the_head_and_matches_what_it_predicts(tmp_path):
    root = _dataset(tmp_path / "data")
    config = _segmenting(root)
    scenes = BreakingBadScenes(config, "train")
    assert scenes.segmenting and all(len(f.token_vertices) == 0 for f in scenes[0].fragments)
    batch, _ = _collate_samples([scenes[0], scenes[1]])
    torch.manual_seed(0)
    model = build_model(config)
    criterion = build_criterion(config)
    model.train()
    total, report, _ = _forward(model, batch, criterion, config)
    assert total.requires_grad and "segment" in report and report["segment"] > 0
    assert {"seg_hit", "seg_chosen", "seg_marked", "dihedral_hit"} <= set(report)
    model.eval()
    keep = {}
    with torch.no_grad():
        _forward(model, batch, criterion, config, keep=keep)
    predicted = torch.sigmoid(keep["prediction"].fracture_logit) > config.segment_threshold
    assert torch.equal(keep["batch"].fracture, predicted)         # what stage two matches
    assert torch.equal(keep["batch"].token_index, keep["prediction"].token_index)
    for scene, candidates in enumerate(keep["matched"].candidates):
        v0, v1 = (int(keep["batch"].vertex_ptr[keep["batch"].fragment_ptr[scene + k]])
                  for k in (0, 1))
        assert torch.equal(candidates, predicted[v0:v1])


def test_the_tokens_come_from_the_truth_when_taught_or_when_the_oracle_asks(tmp_path,
                                                                            monkeypatch):
    from reassembly import training

    root = _dataset(tmp_path / "data")
    config = _segmenting(root, segment_teacher=1.0)
    scenes = BreakingBadScenes(config, "train")
    batch, _ = _collate_samples([scenes[0]])
    torch.manual_seed(0)
    model = build_model(config)
    model.train()
    keep = {}
    monkeypatch.setattr(training.random, "random", lambda: 0.0)
    _forward(model, batch, build_criterion(config), config, keep=keep)
    assert bool(batch.breaks[keep["prediction"].token_index].all())
    # The oracle: in evaluation, the tokens AND the matched vertices are the truth.
    oracle = dataclasses.replace(config, label_method="marked")
    batch, _ = _collate_samples([BreakingBadScenes(oracle, "val")[0]])
    model.eval()
    keep = {}
    with torch.no_grad():
        _forward(model, batch, build_criterion(oracle), oracle, keep=keep)
    assert torch.equal(keep["batch"].fracture, batch.breaks)
    assert bool(batch.breaks[keep["prediction"].token_index].all())


# --------------------------------------------------------------------------
# 4. The consistency term, through the data and the forward pass
# --------------------------------------------------------------------------

@pytest.mark.usefixtures("thinning")
def test_a_remeshed_scene_brings_itself_as_stored_and_the_term_trains(tmp_path):
    root = _dataset(tmp_path / "data")
    config = _config(root, remesh=(0.5, 0.9), remesh_jitter=1e-3, contact_radius=0.35,
                     w_consistency=1.0)
    scenes = BreakingBadScenes(config, "train")
    first, second = scenes[0], scenes[1]
    stored = sum(len(m.vertices) for m in scenes.meshes(0))
    assert first.view is not None and sum(len(f.target_vertices) for f in
                                          first.view.fragments) == stored
    assert len(first.view_index) == sum(len(f.target_vertices) for f in first.fragments)
    assert first.view_index.min() >= 0 and first.view_index.max() < stored
    # Each vertex's stored partner is where it was before the jitter.
    world = np.concatenate([np.asarray(m.vertices) for m in scenes.meshes(0)])
    closed = np.concatenate([np.asarray(m.vertices) for m in scenes._remeshed(
        scenes.key(0), *_pieces(scenes.meshes(0)))[0]])
    assert np.linalg.norm(world[first.view_index] - closed, axis=1).max() < 0.3
    batch, _ = _collate_samples([first, second])
    assert batch.view.num_scenes == 2
    offset = sum(len(f.target_vertices) for f in first.view.fragments)
    tail = batch.view_index[len(first.view_index):]
    assert int(tail.min()) >= offset
    torch.manual_seed(0)
    model = build_model(config)
    model.train()
    total, report, _ = _forward(model, batch, build_criterion(config), config)
    assert "consistency" in report and 0 < report["consistency"] < 2 and total.requires_grad
    # Validation is never augmented, so it brings no view.
    assert BreakingBadScenes(config, "val")[0].view is None


# --------------------------------------------------------------------------
# The whole run, and a v7 run's weights carried into the new schedule
# --------------------------------------------------------------------------

@pytest.mark.usefixtures("thinning")
def test_a_run_with_every_change_trains_validates_and_evaluates(tmp_path, capsys):
    from reassembly import training

    root = _dataset(tmp_path / "data")
    config = _config(root, out_dir=str(tmp_path / "run"),
                     schedule=("intra", "intra", "segment", "global", "cross", "intra"),
                     remesh=(0.5, 0.9), remesh_gap=0.3, remesh_smooth=2, remesh_noise=0.01,
                     remesh_deform=0.01, remesh_jitter=1e-3, augment_probability=0.6,
                     contact_radius=0.35, w_consistency=0.5)
    history = training.train(config)
    out = capsys.readouterr().out
    assert len(history) == 1 and (tmp_path / "run" / "best.pt").exists()
    assert "fracture head x1" in out and "consistency x0.5" in out
    assert "fracture head: precision" in out
    summary = training.evaluate(config, checkpoint="last.pt", split="val")
    assert "seg_precision" in summary and summary["evaluation"]["fracture_head"]
    oracle = dataclasses.replace(config, label_method="marked")
    summary = training.evaluate(oracle, checkpoint="last.pt", split="val",
                                override=("label_method",))
    assert summary["evaluation"]["labels"] == "marked"
    assert "the TRUE break" in capsys.readouterr().out


def test_a_v7_runs_weights_carry_into_a_schedule_with_a_head(tmp_path, capsys):
    root = _dataset(tmp_path / "data")
    old = _config(root, schedule=("intra", "cross", "intra"))
    torch.manual_seed(1)
    trained = build_model(old)
    path = tmp_path / "v7.pt"
    save_checkpoint(path, trained, None, old, epoch=0, step=1, history=[], best=0.0)
    new = _config(root, schedule=("intra", "segment", "cross", "intra", "global"),
                  init_from=str(path))
    with pytest.raises(ValueError, match="--schedule"):
        _start_from(build_model(new), new, "cpu", announce=False)
    partial = dataclasses.replace(new, init_partial=True)
    model = build_model(partial)
    _start_from(model, partial, "cpu", announce=True)
    assert "start fresh -- context.0, fracture_head" in capsys.readouterr().out
    carried = trained.state_dict()
    for key, value in model.state_dict().items():
        if key in carried:
            assert torch.equal(value, carried[key]), key
        else:
            assert key.startswith(("fracture_head.", "context."))


def test_an_old_checkpoint_loads_and_evaluates_as_it_did(tmp_path):
    """A v7 checkpoint has no new modules and no new settings: it loads
    strictly, and its config fills the new fields with the defaults that are
    what it did."""
    from reassembly.training import load_checkpoint

    root = _dataset(tmp_path / "data")
    old = _config(root, schedule=("intra", "cross", "intra"))
    path = tmp_path / "v7.pt"
    save_checkpoint(path, build_model(old), None, old, epoch=0, step=1, history=[], best=0.0)
    state = torch.load(path, weights_only=False)
    for name in ("remesh_edge", "remesh_smooth", "w_segment", "w_consistency",
                 "segment_threshold", "augment_probability"):
        state["config"].pop(name, None)
    torch.save(state, path)
    model, config, _, _ = load_checkpoint(path, Config(root=str(root)))
    assert config.schedule == ("intra", "cross", "intra") and not getattr(model, "segments")
    assert config.remesh_edge is None and config.w_consistency == 0.0
    assert Path(path).exists()

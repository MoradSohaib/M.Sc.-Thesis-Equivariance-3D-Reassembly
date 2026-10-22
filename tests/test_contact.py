"""
Breaks whose two sides do not share their vertices: each piece remeshed on
its own (``data/remesh.py``), the embedding term's positives by distance
(``nn/losses.py``, ``proximity_loss``; ``mesh/correspondence.py``,
``contact_vertices``), and a run started from another run's weights
(``--init_from``).

The model trained on Breaking Bad's coincident break vertices matched scanned
pieces at chance, and Fractura's synthetic bones with each piece thinned on its
own fell from part accuracy 0.659 to 0.047: what it learned was the exact
twin. ``test_the_model_learns_breaks_that_share_no_vertex`` is the check that
the distance-based term fixes that on geometry where no twin exists at all.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reassembly.data.features import build_scene  # noqa: E402
from reassembly.data.remesh import (GAP_REACH, Remesh, break_masks, collapse_pieces,  # noqa: E402
                                    jitter_pieces, open_pieces, remesh_pieces)
from reassembly.data.transforms import random_rotations  # noqa: E402
from reassembly.mesh.correspondence import (compute_scene_correspondence,  # noqa: E402
                                            contact_vertices, find_shared_points)
from reassembly.nn.losses import correspondence_loss, proximity_loss  # noqa: E402
from reassembly.training import (BreakingBadScenes, CHANCE, Config, _adopt_checkpoint_settings,  # noqa: E402
                                 _collate_samples, _forward, _start_from, build_criterion,
                                 build_model, run_epoch, save_checkpoint)

from test_training_recovery import _config, _dataset  # noqa: E402


def _shared(pieces) -> int:
    return sum(len(cluster) for cluster in find_shared_points([v for v, _ in pieces]))


@pytest.fixture
def thinning():
    pytest.importorskip("fast_simplification")


# --------------------------------------------------------------------------
# Remeshing
# --------------------------------------------------------------------------

def _volume(vertices, faces) -> float:
    corners = vertices[faces] - vertices.mean(axis=0)
    return float(np.einsum("ij,ij->i", corners[:, 0],
                           np.cross(corners[:, 1], corners[:, 2])).sum() / 6.0)


@pytest.mark.usefixtures("thinning")
def test_remeshing_takes_the_twins_away(fractured_solid):
    """
    Quadric collapse takes the flattest faces first. The fixture's far sides
    are flat and its interface is noise, so the interface goes last: at 30% to
    50% of the faces kept, about half of its twins are left (622 of 1,152 here;
    at 50% to 90%, 1,126). The jitter takes the rest.
    """
    vertices = [np.asarray(m.vertices, float) for m in fractured_solid]
    faces = [np.asarray(m.faces) for m in fractured_solid]
    before = _shared(list(zip(vertices, faces)))
    assert before == 2 * 24 * 24                          # the interface, on both sides
    v, f = remesh_pieces(vertices, faces, np.random.default_rng(0), Remesh(0.3, 0.5))
    for f0, f1 in zip(faces, f):
        assert 0.98 * 0.3 * len(f0) <= len(f1) <= 0.5 * len(f0) + 1
    assert 0 < _shared(list(zip(v, f))) < 0.75 * before
    moved, same = remesh_pieces(vertices, faces, np.random.default_rng(0),
                                Remesh(0.3, 0.5, jitter=1e-3))
    assert _shared(list(zip(moved, same))) == 0
    # The jitter moves the vertices, not the tessellation: the shares are
    # drawn before it.
    assert all(np.array_equal(a, b) for a, b in zip(f, same))
    # Still the same solids, a little coarser, wound as they were -- a piece
    # wound inward too, which the tidy alone would turn outward.
    for v0, f0, v1, f1 in zip(vertices, faces, moved, same):
        assert np.allclose(np.ptp(v1, axis=0), np.ptp(v0, axis=0), atol=0.01)
        assert _volume(v0, f0) > 0 and _volume(v1, f1) > 0
    inward = [np.ascontiguousarray(f0[:, ::-1]) for f0 in faces]
    v, f = remesh_pieces(vertices, inward, np.random.default_rng(0), Remesh(0.3, 0.5))
    assert all(_volume(v1, f1) < 0 for v1, f1 in zip(v, f))


@pytest.mark.usefixtures("thinning")
def test_remeshing_is_repeatable_and_draws_each_piece_its_share(fractured_solid):
    vertices = [np.asarray(m.vertices, float) for m in fractured_solid]
    faces = [np.asarray(m.faces) for m in fractured_solid]
    spec = Remesh(0.4, 0.9, jitter=1e-3)
    first = remesh_pieces(vertices, faces, np.random.default_rng(3), spec)
    again = remesh_pieces(vertices, faces, np.random.default_rng(3), spec)
    other = remesh_pieces(vertices, faces, np.random.default_rng(4), spec)
    assert all(np.array_equal(a, b) for a, b in zip(first[0] + first[1], again[0] + again[1]))
    assert not all(len(a) == len(b) and np.allclose(a, b) for a, b in zip(first[0], other[0]))
    # The collapse, then the jitter, from one generator: what the dataset does
    # to label the pieces between the two.
    rng = np.random.default_rng(3)
    collapsed, kept = collapse_pieces(vertices, faces, rng, spec)
    moved = jitter_pieces(collapsed, rng, spec.jitter)
    assert all(np.array_equal(a, b) for a, b in zip(moved + kept, first[0] + first[1]))
    # With an opening, it comes between the two, on the breaks the pieces share.
    opened = Remesh(0.4, 0.9, jitter=1e-3, gap=0.3)
    rng = np.random.default_rng(3)
    collapsed, kept = collapse_pieces(vertices, faces, rng, opened)
    moved = jitter_pieces(open_pieces(collapsed, kept, break_masks(vertices, collapsed), rng,
                                      opened.gap), rng, opened.jitter)
    whole = remesh_pieces(vertices, faces, np.random.default_rng(3), opened)
    assert all(np.array_equal(a, b) for a, b in zip(moved + kept, whole[0] + whole[1]))
    assert all(np.array_equal(a, b) for a, b in zip(kept, first[1]))
    assert np.abs(np.concatenate(whole[0]) - np.concatenate(first[0])).max() > 0.01
    # A piece too small to collapse keeps its faces, and the jitter still moves it.
    tiny_v = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    tiny_f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    v, f = remesh_pieces([tiny_v, vertices[1]], [tiny_f, faces[1]], np.random.default_rng(3),
                         Remesh(0.4, 0.9))
    assert np.array_equal(f[0], tiny_f) and np.array_equal(v[0], tiny_v)
    moved, _ = remesh_pieces([tiny_v], [tiny_f], np.random.default_rng(3), spec)
    assert not np.array_equal(moved[0], tiny_v) and np.abs(moved[0] - tiny_v).max() < 0.01
    # ... and the piece after it collapses as it does after a piece of any size.
    plain = remesh_pieces(vertices, faces, np.random.default_rng(3), Remesh(0.4, 0.9))
    assert np.array_equal(f[1], plain[1][1]) and np.array_equal(v[1], plain[0][1])


@pytest.mark.usefixtures("thinning")
def test_remeshing_merges_the_vertices_the_collapse_leaves_together(fractured_solid):
    """
    On 96 Breaking Bad pieces the collapse left 436 vertex pairs within 1e-6
    of the piece's diagonal, 1e-8 to 1.3e-6 apart, the faces between them
    slivers. No synthetic mesh tried makes it do that, so a vertex split in
    two a billionth apart stands in for them: the tidy merges them, and the
    piece is closed again.
    """
    from scipy.spatial import cKDTree

    from reassembly.mesh.clean import edge_counts

    lower = fractured_solid[0]
    vertices, faces = np.asarray(lower.vertices, float), np.asarray(lower.faces).copy()
    centre = 12 * 24 + 12                                  # mid-interface
    fan = np.flatnonzero((faces == centre).any(axis=1))[:3]
    vertices = np.vstack([vertices, vertices[centre] + [1e-9, 0.0, 0.0]])
    faces[fan] = np.where(faces[fan] == centre, len(vertices) - 1, faces[fan])
    assert edge_counts(faces)[0] > 0
    v, f = remesh_pieces([vertices], [faces], np.random.default_rng(0), Remesh(0.5, 0.9))
    diagonal = float(np.linalg.norm(np.ptp(v[0], axis=0)))
    assert not len(cKDTree(v[0]).query_pairs(1e-6 * diagonal))
    assert edge_counts(f[0])[0] == 0


@pytest.mark.usefixtures("thinning")
def test_remeshing_keeps_every_component_and_spreads_the_shares(fractured_solid):
    import trimesh

    from reassembly.mesh.clean import face_components

    lower = fractured_solid[0]
    vertices, faces = np.asarray(lower.vertices, float), np.asarray(lower.faces)
    # An island of 80 faces, 0.6% of the piece's area: the converter's debris
    # rule would drop it from a scan, the Breaking Bad reader keeps it, and so
    # does remeshing.
    ball = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
    island_v = np.vstack([vertices, np.asarray(ball.vertices) + [0.0, 0.0, -3.0]])
    island_f = np.vstack([faces, np.asarray(ball.faces) + len(vertices)])
    v, f = remesh_pieces([island_v], [island_f], np.random.default_rng(0), Remesh(0.5, 0.9))
    assert face_components(f[0], len(v[0]))[0] == 2
    # The shares are drawn from the whole of [LOW, HIGH].
    kept = [len(remesh_pieces([vertices], [faces], np.random.default_rng(seed),
                              Remesh(0.3, 0.9))[1][0]) / len(faces) for seed in range(12)]
    assert min(kept) < 0.45 and max(kept) > 0.75


def test_remesh_settings_are_checked():
    for bad in ((0.0, 0.5), (0.6, 0.5), (0.5, 1.1)):
        with pytest.raises(ValueError, match="0 < LOW <= HIGH <= 1"):
            Config(remesh=bad)
    with pytest.raises(ValueError, match="give --remesh"):
        Config(remesh_jitter=1e-3)
    with pytest.raises(ValueError, match="two numbers"):
        Config(remesh=(0.5,))
    with pytest.raises(ValueError, match="contact_band"):
        Config(contact_radius=0.01, contact_band=0.5)
    assert Config(remesh=[0.5, 0.9]).remesh == (0.5, 0.9)
    with pytest.raises(ValueError, match="give --remesh"):
        Config(remesh_gap=0.03)
    with pytest.raises(ValueError, match="remesh_gap is the largest mean opening"):
        Config(remesh=(0.5, 0.9), remesh_gap=-0.01)


# --------------------------------------------------------------------------
# Opening the breaks
# --------------------------------------------------------------------------

def _halves(heights, n: int = 24, side: float = 0.5, depth: float = 0.4):
    """Two closed slabs meeting on a break of the given heights (``n x n``),
    ``side`` across -- 0.5 at 24 a side is an edge of 0.022, about Breaking
    Bad's -- the lower one's break on top."""
    from conftest import _build_half

    x = np.linspace(-side / 2, side / 2, n)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    interface = np.stack([gx.ravel(), gy.ravel(), np.ravel(heights)], axis=1)
    halves = (_build_half(interface, n, -depth, upward=True),
              _build_half(interface, n, depth, upward=False))
    return [np.asarray(m.vertices, float) for m in halves], [np.asarray(m.faces) for m in halves]


def _turned(before, after, faces) -> int:
    def normals(v):
        return np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    return int((np.einsum("ij,ij->i", normals(before), normals(after)) <= 0).sum())


def test_opening_moves_each_side_of_the_break_along_its_normal():
    """
    A flat break: each side's break vertices move along z and nothing else
    does. Over 16 draws the two sides stand 1.04 times the drawn mean opening
    apart (0.94 to 1.18 per draw), and in every draw 8.3% or more of each side
    moves outward, into the other piece, and 12.0% or more inward by over
    twice its mean.
    """
    vertices, faces = _halves(np.zeros(24 * 24))
    breaks = break_masks(vertices, vertices)
    assert all(b[:24 * 24].all() and not b[24 * 24:].any() for b in breaks)
    apart = []
    for seed in range(16):
        opened = open_pieces(vertices, faces, breaks, np.random.default_rng(seed), 0.03)
        mean = 0.03 * np.random.default_rng(seed).uniform()      # the scene's draw, first
        inward = []
        for sign, v0, v1, b in zip((-1, 1), vertices, opened, breaks):
            moved = v1 - v0
            assert not moved[:, :2].any() and not moved[~b].any()
            inward.append(sign * moved[b, 2])
            assert (inward[-1] < 0).mean() > 0.08 and (inward[-1] > mean).mean() > 0.1
        apart.append((inward[0].mean() + inward[1].mean()) / mean)
        assert not np.allclose(inward[0], inward[1])              # a field per side
    assert abs(np.mean(apart) - 1.0) < 0.1
    # Repeatable; without a gap, nothing moves and nothing is drawn.
    again = open_pieces(vertices, faces, breaks, np.random.default_rng(15), 0.03)
    assert all(np.array_equal(a, b) for a, b in zip(again, opened))
    rng = np.random.default_rng(3)
    assert all(np.array_equal(a, b) for a, b in
               zip(open_pieces(vertices, faces, breaks, rng, 0.0), vertices))
    assert rng.uniform() == np.random.default_rng(3).uniform()
    # The draws do not depend on the masks: a piece with no break stays, and
    # the other's field is the one it would have had.
    rng, other = np.random.default_rng(4), np.random.default_rng(4)
    one = open_pieces(vertices, faces, [np.zeros_like(breaks[0]), breaks[1]], rng, 0.03)
    both = open_pieces(vertices, faces, breaks, other, 0.03)
    assert np.array_equal(one[0], vertices[0]) and np.array_equal(one[1], both[1])
    assert rng.uniform() == other.uniform()


def test_the_vertices_beside_a_break_move_with_it():
    """Half of a flat face is the break: the rest of the face moves as its
    nearest break vertex does, by a smoothstep weight falling from 1 at the
    break to 0 at GAP_REACH times that vertex's step; the far side stays."""
    from scipy.spatial import cKDTree

    vertices, faces = _halves(np.zeros(24 * 24))
    v0, f = vertices[0], faces[0]
    top = np.zeros(len(v0), bool)
    top[:24 * 24] = True
    mask = top & (v0[:, 0] < 0)
    for seed in range(3):
        (v1,) = open_pieces([v0], [f], [mask], np.random.default_rng(seed), 0.03)
        moved = v1 - v0
        assert not moved[:, :2].any() and not moved[~top].any()
        distance, nearest = cKDTree(v0[mask]).query(v0)
        step = moved[mask][nearest, 2]
        u = np.clip(distance / (GAP_REACH * np.abs(step)), 0.0, 1.0)
        assert np.allclose(moved[:, 2], (1.0 - u * u * (3.0 - 2.0 * u)) * step, atol=1e-15)
        assert (moved[top & ~mask, 2] != 0).any() and (moved[top & ~mask, 2] == 0).any()


def test_an_opened_break_turns_no_face_over():
    """
    A rough break (heights 0.02 against an edge of 0.022), opened by up to
    0.1: moved as asked, 164 faces of the two pieces would turn over at this
    draw; cut back where they would, none does, and the break still opens.
    """
    import reassembly.data.remesh as remesh

    heights = np.random.default_rng(7).normal(scale=0.02, size=(24, 24))
    heights[0, :] = heights[-1, :] = heights[:, 0] = heights[:, -1] = 0.0
    vertices, faces = _halves(heights)
    breaks = break_masks(vertices, vertices)
    opened = open_pieces(vertices, faces, breaks, np.random.default_rng(0), 0.1)
    for v0, v1, f, b in zip(vertices, opened, faces, breaks):
        assert _turned(v0, v1, f) == 0
        assert np.linalg.norm(v1 - v0, axis=1)[b].max() > 0.1
    asked = remesh._unturned
    try:
        remesh._unturned = lambda vertices, faces, step: step
        unguarded = open_pieces(vertices, faces, breaks, np.random.default_rng(0), 0.1)
    finally:
        remesh._unturned = asked
    assert sum(_turned(v0, v1, f) for v0, v1, f in zip(vertices, unguarded, faces)) == 164
    # A face of no area has no side to turn to: left out, not cut back for ever.
    (v1,) = open_pieces([vertices[0]], [np.vstack([faces[0], [[0, 0, 1]]])], [breaks[0]],
                        np.random.default_rng(0), 0.1)
    assert np.array_equal(v1, opened[0])


@pytest.mark.usefixtures("thinning")
def test_the_break_is_carried_over_to_the_collapse(fractured_solid):
    vertices = [np.asarray(m.vertices, float) for m in fractured_solid]
    faces = [np.asarray(m.faces) for m in fractured_solid]
    collapsed, _ = collapse_pieces(vertices, faces, np.random.default_rng(0), Remesh(0.5, 0.9))
    for v, b in zip(collapsed, break_masks(vertices, collapsed)):
        # The interface (heights of noise 0.08) is the break, the far side
        # (z = -1.5 or 1.5) is not.
        assert b.any() and (np.abs(v[b, 2]) < 0.5).all() and (np.abs(v[~b, 2]) > 0.5).all()


# --------------------------------------------------------------------------
# Contact, and the distance-based term
# --------------------------------------------------------------------------

def test_contact_is_another_pieces_vertex_within_the_radius():
    # A line of vertices 0.05 apart, and three of another piece 0.07 above its
    # first three: the fourth is 0.086 from the nearest of them. A piece's own
    # vertices, 0.05 apart, are within either radius and never count.
    line = np.stack([np.arange(21) * 0.05, np.zeros(21), np.zeros(21)], 1)
    other = line[:3] + [0.0, 0.07, 0.0]
    a, b = contact_vertices([line, other], radius=0.08)
    assert a.tolist() == [True] * 3 + [False] * 18 and b.all()
    a, b = contact_vertices([line, other], radius=0.06)
    assert not a.any() and not b.any()
    a, = contact_vertices([line], radius=1.0)               # one piece: nothing to touch
    assert not a.any()


def _embedded(points, groups, scenes, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    z = torch.tensor(rng.standard_normal((len(points), dim)), dtype=torch.float64)
    return (z, torch.tensor(points, dtype=torch.float64), torch.tensor(groups),
            torch.tensor(scenes))


def test_the_term_counts_the_band_as_neither():
    """Four anchors on a line, radius 1, band 2: a and b are each other's
    positive; c, 1.5 from a and 1 from b on b's own piece, is ignored by both
    and has no positive of its own; d, far off, is a negative for everyone."""
    points = [[0, 0, 0], [0.5, 0, 0], [1.5, 0, 0], [10, 0, 0]]
    z, p, g, s = _embedded(points, [0, 1, 1, 0], [0, 0, 0, 0])
    loss, accuracy = proximity_loss(z, p, g, s, radius=1.0, band=2.0, return_accuracy=True)
    unit = torch.nn.functional.normalize(z, dim=-1)
    sim = unit @ unit.t() / 0.1
    expected = 0.5 * ((torch.logsumexp(sim[0, [1, 3]], 0) - sim[0, 1])
                      + (torch.logsumexp(sim[1, [0, 3]], 0) - sim[1, 0]))
    assert torch.isclose(loss, expected)
    hits = [int(torch.argmax(sim[0, [1, 3]]) == 0), int(torch.argmax(sim[1, [0, 3]]) == 0)]
    assert float(accuracy) == pytest.approx(sum(hits) / 2)


def test_the_criterion_takes_its_radius_and_band():
    from reassembly.nn.losses import ReassemblyLoss

    points = [[0, 0, 0], [0.5, 0, 0], [1.5, 0, 0], [10, 0, 0]]
    z, p, g, s = _embedded(points, [0, 1, 1, 0], [0, 0, 0, 0])
    values = []
    for radius, band in ((1.0, 2.0), (1.0, 1.0), (2.0, 1.5)):
        total, report = ReassemblyLoss(contact_radius=radius, contact_band=band)(
            embeddings=z, positions=p, groups=g, scenes=s)
        assert torch.isclose(total, proximity_loss(z, p, g, s, radius, band=band))
        values.append(float(total))
    assert len(set(values)) == 3
    # Given clusters and no positions, the coincidence term.
    cluster = torch.tensor([0, 0, -1, -1])
    total, _ = ReassemblyLoss(contact_radius=1.0)(embeddings=z, cluster=cluster, num_clusters=1)
    assert torch.isclose(total, correspondence_loss(z, cluster, 1))
    # Training builds it from the run's settings.
    criterion = build_criterion(Config(contact_radius=0.5, contact_band=3.0))
    assert (criterion.contact_radius, criterion.contact_band) == (0.5, 3.0)


def test_another_scene_is_neither_positive_nor_negative():
    """Two scenes at the very same place, three anchors each: an anchor's one
    positive is its own scene's other piece, its negative the far vertex of its
    own scene -- the other scene's vertices, coincident or not, are not in its
    term at all. Matching never compares two scenes, so they would teach
    nothing; and the term does not change when its scenes share a pass."""
    points = [[0, 0, 0], [0.1, 0, 0], [5, 0, 0]] * 2
    z, p, g, s = _embedded(points, [0, 1, 0, 2, 3, 2], [0, 0, 0, 1, 1, 1])
    loss = proximity_loss(z, p, g, s, radius=0.5)
    unit = torch.nn.functional.normalize(z, dim=-1)
    sim = unit @ unit.t() / 0.1
    terms = [torch.logsumexp(sim[i, others], 0) - sim[i, j]
             for i, j, others in ((0, 1, [1, 2]), (1, 0, [0, 2]), (3, 4, [4, 5]), (4, 3, [3, 5]))]
    assert torch.isclose(loss, torch.stack(terms).mean())
    first = proximity_loss(z[:3], p[:3], g[:3], s[:3], radius=0.5)
    second = proximity_loss(z[3:], p[3:], g[3:], s[3:], radius=0.5)
    assert torch.isclose(loss, (first + second) / 2)


def test_each_scene_of_a_pass_draws_its_own_anchors():
    """The budget is per scene: a scene sharing a pass with a large one keeps
    every anchor it would have had alone (here all of them), where one budget
    for the pass left it a share -- and an anchor trains only when one of its
    positives is drawn with it."""
    from reassembly.nn.losses import _scene_anchors

    scenes = torch.tensor([0] * 3000 + [1] * 40)
    drawn = dict(_scene_anchors(len(scenes), scenes, 1024, None, scenes.device))
    assert len(drawn[0]) == 1024 and len(drawn[1]) == 40
    assert bool((scenes[drawn[1]] == 1).all()) and bool((scenes[drawn[0]] == 0).all())
    # One scene: the draw it always was.
    torch.manual_seed(3)
    alone = dict(_scene_anchors(3000, None, 1024, None, scenes.device))[0]
    torch.manual_seed(3)
    assert torch.equal(alone, torch.randperm(3000)[:1024])


def test_scene_weights_weight_each_scene_s_mean():
    points = [[0, 0, 0], [0.1, 0, 0], [5, 0, 0]] * 2
    z, p, g, s = _embedded(points, [0, 1, 0, 2, 3, 2], [0, 0, 0, 1, 1, 1])
    first = proximity_loss(z[:3], p[:3], g[:3], s[:3], radius=0.5)
    second = proximity_loss(z[3:], p[3:], g[3:], s[3:], radius=0.5)
    weighted = proximity_loss(z, p, g, s, radius=0.5,
                              scene_weights=torch.tensor([1.0, 3.0], dtype=torch.float64))
    assert torch.isclose(weighted, (first + 3 * second) / 4)
    cluster = torch.tensor([0, 0, -1, 1, 1, -1])
    pair = correspondence_loss(z, cluster, 2, scenes=s,
                               scene_weights=torch.tensor([1.0, 3.0], dtype=torch.float64))
    alone = [correspondence_loss(z[i:i + 3], cluster[i:i + 3] - (1 if i else 0), 1)
             for i in (0, 3)]
    assert torch.isclose(pair, (alone[0] + 3 * alone[1]) / 4)


def test_with_exact_twins_and_no_band_it_is_the_coincidence_term():
    rng = np.random.default_rng(1)
    twins = rng.uniform(0, 1, (20, 3))
    points = np.concatenate([twins, twins])            # piece 0, then piece 1
    groups = [0] * 20 + [1] * 20
    z, p, g, s = _embedded(points, groups, [0] * 40, dim=6)
    near = proximity_loss(z, p, g, s, radius=1e-9, band=1.0)
    exact = correspondence_loss(z, torch.tensor(list(range(20)) * 2), 20)
    assert torch.isclose(near, exact)


def test_matching_embeddings_score_low_and_match():
    points = [[0, 0, 0], [0.01, 0, 0], [1, 0, 0], [1.01, 0, 0], [0, 1, 0], [0, 1.01, 0]]
    codes = torch.eye(3, dtype=torch.float64).repeat_interleave(2, dim=0)   # each pair one code
    p, g, s = (torch.tensor(points, dtype=torch.float64), torch.tensor([0, 1] * 3),
               torch.zeros(6, dtype=torch.long))
    good, hit = proximity_loss(codes, p, g, s, radius=0.05, return_accuracy=True)
    bad, miss = proximity_loss(codes[torch.tensor([0, 2, 4, 1, 3, 5])], p, g, s, radius=0.05,
                               return_accuracy=True)
    assert float(hit) == 1.0 and float(miss) == 0.0 and good < bad


# --------------------------------------------------------------------------
# The dataset and the forward pass
# --------------------------------------------------------------------------

@pytest.mark.usefixtures("thinning")
def test_only_the_training_scenes_are_remeshed(tmp_path):
    root = _dataset(tmp_path / "data")
    config = _config(root, remesh=(0.5, 0.9), remesh_jitter=1e-3, contact_radius=0.35)
    train = BreakingBadScenes(config, "train")
    val = BreakingBadScenes(config, "val")
    assert train.remesh == Remesh(0.5, 0.9, 1e-3) and val.remesh is None

    stored = train.meshes(0)
    sample = train.build(0)
    faces = [len(f.target_edge_normals) for f in sample.fragments]
    plain = BreakingBadScenes(_config(root, contact_radius=0.35), "train").build(0)
    assert sum(faces) < sum(len(f.target_edge_normals) for f in plain.fragments)
    assert [len(m.faces) for m in train.pieces(0)] != [len(m.faces) for m in stored]
    # The same epoch remeshes the same way; another epoch afresh.
    assert all(np.array_equal(a.faces, b.faces) for a, b in zip(train.pieces(0), train.pieces(0)))
    later = BreakingBadScenes(config, "train", epoch_seed=1)
    assert not all(np.array_equal(a.vertices, b.vertices)
                   for a, b in zip(train.pieces(0), later.pieces(0)))
    # What `pieces` holds is what the model was given.
    for piece, fragment in zip(train.pieces(0), sample.fragments):
        assert np.allclose(piece.vertices, fragment.target_vertices * fragment.divisor
                           + fragment.centroid, atol=1e-5)
    # Validation is the scenes as stored.
    assert all(np.array_equal(np.asarray(a.faces), np.asarray(b.faces))
               for a, b in zip(val.pieces(0), val.meshes(0)))
    # An evaluation asks for its own remeshing.
    asked = BreakingBadScenes(config, "val", remesh=Remesh(0.5, 0.5))
    assert asked.remesh == Remesh(0.5, 0.5)
    assert sample.contact is not None and sample.contact.dtype == bool and sample.contact.any()
    assert len(sample.contact) == sum(len(f.target_vertices) for f in sample.fragments)


@pytest.mark.usefixtures("thinning")
def test_the_noise_moves_the_vertices_not_the_fracture_labels(tmp_path):
    """
    The dihedral test reads the jitter as fracture: on seven Breaking Bad
    scenes it flags 13-76% of the vertices as stored, 14-80% after the
    collapse and 90-99.5% after jitter 0.001 -- the labels that place the
    cross-attention tokens and bound stage two's matching. So they are
    computed before the jitter. Here 0.02, about the share of this fixture's
    edge (0.3) that 0.001 is of Breaking Bad's.
    """
    root = _dataset(tmp_path / "data")
    noisy = BreakingBadScenes(_config(root, remesh=(0.5, 0.9), remesh_jitter=0.02,
                                      contact_radius=0.35), "train")
    quiet = BreakingBadScenes(_config(root, remesh=(0.5, 0.9), contact_radius=0.35), "train")
    moved, still = noisy.build(0), quiet.build(0)
    assert all(np.array_equal(a.fracture, b.fracture)
               for a, b in zip(moved.fragments, still.fragments))
    assert not all(np.allclose(a.target_vertices, b.target_vertices)
                   for a, b in zip(moved.fragments, still.fragments))
    # Labelled after the jitter, the same pieces would be most of the surface.
    after = np.concatenate(noisy.labels(noisy.pieces(0)))
    before = np.concatenate([a.fracture for a in moved.fragments])
    assert after.mean() > before.mean() + 0.2


@pytest.mark.usefixtures("thinning")
def test_settings_that_would_train_on_nothing_are_refused(tmp_path):
    from reassembly import training

    root = _dataset(tmp_path / "data")
    # Remeshed breaks have no coincident vertex to label them by.
    with pytest.raises(ValueError, match="coincidence finds a break"):
        BreakingBadScenes(_config(root, remesh=(0.5, 0.9), label_method="coincidence"), "train")
    BreakingBadScenes(_config(root, label_method="coincidence"), "train")
    # ... nor, jittered, a twin for the coincidence term.
    with pytest.raises(ValueError, match="give --contact_radius"):
        training.train(_config(root, out_dir=str(tmp_path / "run"), remesh=(0.5, 0.9),
                               remesh_jitter=1e-3))
    # Without the jitter a share of the twins is left, and the term trains on those.
    remeshed = BreakingBadScenes(_config(root, remesh=(0.5, 0.9)), "train")
    assert (remeshed.build(0).cluster >= 0).any()
    # Opened, no twin is left either.
    with pytest.raises(ValueError, match="give --contact_radius"):
        training.train(_config(root, out_dir=str(tmp_path / "run"), remesh=(0.5, 0.9),
                               remesh_gap=0.5))


@pytest.mark.usefixtures("thinning")
def test_an_opened_scene_finds_its_positives_where_the_pieces_met(tmp_path):
    root = _dataset(tmp_path / "data")
    opened = BreakingBadScenes(_config(root, remesh=(0.5, 0.9), remesh_gap=0.5,
                                       contact_radius=0.35), "train")
    closed = BreakingBadScenes(_config(root, remesh=(0.5, 0.9), contact_radius=0.35), "train")
    assert opened.remesh == Remesh(0.5, 0.9, 0.0, 0.5)
    stored = opened.meshes(0)
    collapsed, surface, seen = opened._remeshed(
        opened.key(0), [np.asarray(m.vertices, float) for m in stored],
        [np.asarray(m.faces) for m in stored])
    sample, plain = opened.build(0), closed.build(0)
    before = np.concatenate([m.vertices for m in collapsed])
    # Collapsed as without the opening; the model reads the opened pieces,
    # opened as `remesh_pieces` opens them -- on the stored pieces' breaks ...
    assert all(np.array_equal(a.vertices, b.vertices) for a, b in zip(collapsed, closed.pieces(0)))
    assert all(np.array_equal(a.vertices, b.vertices) for a, b in zip(seen, opened.pieces(0)))
    from reassembly.training import stable_seed

    rng = np.random.default_rng(stable_seed(opened.key(0), opened.epoch_seed, opened.config.seed,
                                            "remesh"))
    expected, _ = remesh_pieces([np.asarray(m.vertices, float) for m in stored],
                                [np.asarray(m.faces) for m in stored], rng, opened.remesh)
    assert all(np.array_equal(a.vertices, b) for a, b in zip(seen, expected))
    world = np.concatenate(_world(sample))
    assert np.allclose(world, np.concatenate([m.vertices for m in seen]), atol=1e-5)
    assert np.abs(world - before).max() > 0.05
    # ... labelled on the opened surface (no jitter here: what it reads) ...
    assert all(np.array_equal(f.fracture, mask)
               for f, mask in zip(sample.fragments, opened.labels(surface)))
    # ... and its positives are where the pieces met: the same anchors as
    # closed, found among the vertices where they were before the opening.
    assert np.array_equal(sample.contact_points, before) and plain.contact_points is not None
    assert np.array_equal(sample.contact, plain.contact)
    # Not remeshed, a scene leaves the positives to where its vertices are.
    assert BreakingBadScenes(_config(root, contact_radius=0.35), "train").build(0).contact_points is None


def test_the_checkpoints_training_remesh_is_not_an_evaluations():
    stored = dict(Config(remesh=(0.5, 0.9), remesh_jitter=1e-3, remesh_gap=0.03,
                         contact_radius=0.015).__dict__)
    adopted = _adopt_checkpoint_settings(Config(), stored, data=True, quiet=True)
    assert adopted.remesh is None and adopted.remesh_jitter == 0.0 and adopted.remesh_gap == 0.0
    assert adopted.contact_radius == 0.015                # the term is the checkpoint's


def _world(sample):
    return [f.target_vertices * f.divisor + f.centroid for f in sample.fragments]


def _contact_batch(radius):
    from test_training import _scene

    samples = []
    for seed in (1, 2):
        sample = _scene(seed)
        samples.append(sample._replace(
            contact=np.concatenate(contact_vertices(_world(sample), radius))))
    batch, _ = _collate_samples(samples)
    return batch, samples


class _Seen(torch.nn.Module):
    """The criterion, keeping what it was called with."""

    def __init__(self, criterion):
        super().__init__()
        self.criterion, self.seen = criterion, {}

    def forward(self, *args, **kwargs):
        self.seen = kwargs
        return self.criterion(*args, **kwargs)


def test_the_forward_pass_trains_on_the_contact_vertices():
    torch.manual_seed(0)
    config = Config(channels=32, heads=4, contact_radius=1.0)
    batch, samples = _contact_batch(1.0)              # the shards are 2.5 apart, radius 1
    assert batch.contact is not None and bool(batch.contact.any())
    model = build_model(config)
    criterion = _Seen(build_criterion(config))
    loss, report, _ = _forward(model, batch, criterion, config)
    assert loss.requires_grad and torch.isfinite(loss) and "match@1" in report
    # The anchors are the contact vertices, where they lie assembled, in world
    # units, with their pieces and scenes numbered as the batch numbers them.
    fragments = [f for sample in samples for f in sample.fragments]
    contact = np.concatenate([sample.contact for sample in samples])
    world = np.concatenate([p for sample in samples for p in _world(sample)])
    piece = np.repeat(np.arange(len(fragments)), [len(f.target_vertices) for f in fragments])
    scene = np.repeat(np.arange(len(samples)),
                      [sum(len(f.target_vertices) for f in s.fragments) for s in samples])
    seen = criterion.seen
    assert np.allclose(seen["positions"].numpy(), world[contact], atol=1e-5)
    assert np.array_equal(seen["groups"].numpy(), piece[contact])
    assert np.array_equal(seen["scenes"].numpy(), scene[contact])
    assert seen["embeddings"].shape[0] == contact.sum() and seen["cluster"] is None
    # The same batch through the coincidence term is another number.
    plain = Config(channels=32, heads=4)
    other, _, _ = _forward(model, batch, build_criterion(plain), plain)
    assert not torch.isclose(loss, other)
    # No vertex in contact: nothing to train on, as with no cluster.
    far, _ = _contact_batch(1e-6)
    assert not bool(far.contact.any())
    loss, _, _ = _forward(model, far, build_criterion(config), config)
    assert not loss.requires_grad


def test_the_positives_are_found_where_the_scene_says_the_vertices_were():
    """An opened scene carries where its vertices were before the opening
    (``contact_points``); the term finds the positives there, not where the
    model's vertices are. A scene in the batch without them brings the
    latter."""
    from test_training import _scene

    torch.manual_seed(0)
    config = Config(channels=32, heads=4, contact_radius=1.0)
    samples = []
    for seed in (1, 2):
        sample = _scene(seed)
        world = _world(sample)
        samples.append(sample._replace(contact=np.concatenate(contact_vertices(world, 1.0)),
                                       contact_points=np.concatenate(world) + 0.5))
    batch, _ = _collate_samples(samples)
    criterion = _Seen(build_criterion(config))
    _forward(build_model(config), batch, criterion, config)
    contact = np.concatenate([s.contact for s in samples])
    points = np.concatenate([s.contact_points for s in samples])
    assert np.allclose(criterion.seen["positions"].numpy(), points[contact], atol=1e-5)
    mixed, _ = _collate_samples([samples[0], samples[1]._replace(contact_points=None)])
    expected = np.concatenate([samples[0].contact_points, np.concatenate(_world(samples[1]))])
    assert np.allclose(mixed.contact_points.numpy(), expected, atol=1e-5)
    assert _collate_samples([s._replace(contact_points=None) for s in samples])[0].contact_points is None
    meshes, breaks = _unshared(0, 2)
    with pytest.raises(ValueError, match=r"contact_points is \(3, 3\) for 362 vertices"):
        build_scene([np.asarray(m.vertices) for m in meshes], [np.asarray(m.faces) for m in meshes],
                    breaks, contact_points=np.zeros((3, 3)))


def test_a_scene_without_the_mask_brings_no_anchors():
    from test_training import _scene

    marked, unmarked = _scene(1), _scene(2)
    mask = np.zeros(sum(len(f.target_vertices) for f in marked.fragments), bool)
    mask[:5] = True
    batch, _ = _collate_samples([marked._replace(contact=mask), unmarked])
    total = int(batch.target_vertices.shape[0])
    assert batch.contact.tolist() == mask.tolist() + [False] * (total - len(mask))
    assert _collate_samples([unmarked])[0].contact is None
    meshes, breaks = _unshared(0, 2)
    with pytest.raises(ValueError, match="contact has 3 entries for 362 vertices"):
        build_scene([np.asarray(m.vertices) for m in meshes], [np.asarray(m.faces) for m in meshes],
                    breaks, contact=np.zeros(3, bool))


# --------------------------------------------------------------------------
# --init_from
# --------------------------------------------------------------------------

def test_a_new_run_starts_from_the_weights_alone(tmp_path, capsys):
    torch.manual_seed(0)
    source = Config(channels=16, heads=4, head_dim=4, embedding_dim=8)
    trained = build_model(source)
    state = trained.state_dict()
    path = tmp_path / "best.pt"
    save_checkpoint(path, trained, None, source, epoch=4, step=10, history=[], best=0.5)
    torch.manual_seed(2)
    save_checkpoint(tmp_path / "last.pt", build_model(source), None, source, epoch=5, step=12,
                    history=[], best=0.5)

    # A folder is its best.pt.
    torch.manual_seed(1)
    fresh = build_model(source)
    _start_from(fresh, Config(channels=16, heads=4, head_dim=4, embedding_dim=8,
                              init_from=str(tmp_path)), "cpu", announce=True)
    assert all(torch.equal(state[k], v) for k, v in fresh.state_dict().items())
    assert f"started from the weights of {path} (epoch 5)" in capsys.readouterr().out

    # A pre-v7 checkpoint's rotation head stays behind.
    with_head = dict(state, **{"head.weight": torch.zeros(3, 3)})
    torch.save({"model": with_head, "config": dict(source.__dict__), "epoch": 0}, path)
    _start_from(build_model(source), Config(channels=16, heads=4, head_dim=4, embedding_dim=8,
                                            init_from=str(path)), "cpu", announce=True)
    assert "leaving the rotation head" in capsys.readouterr().out

    with pytest.raises(ValueError, match="Pass --hidden_channels 16"):
        _start_from(build_model(Config(channels=32, heads=4, head_dim=4, embedding_dim=8)),
                    Config(channels=32, heads=4, head_dim=4, embedding_dim=8,
                           init_from=str(path)), "cpu", announce=False)
    with pytest.raises(FileNotFoundError):
        _start_from(fresh, Config(init_from=str(tmp_path / "missing.pt")), "cpu", False)


def test_training_starts_from_init_from_and_resumes_its_own(tmp_path, capsys):
    from reassembly import training

    root = _dataset(tmp_path / "data")
    first = _config(root, out_dir=str(tmp_path / "first"), steps_per_epoch=1)
    training.train(first)
    capsys.readouterr()
    second = _config(root, out_dir=str(tmp_path / "second"), steps_per_epoch=1,
                     init_from=str(tmp_path / "first" / "last.pt"), contact_radius=0.35)
    history = training.train(second)
    out = capsys.readouterr().out
    assert "starting from --init_from's weights" in out and "started from the weights" in out
    assert len(history) == 1 and "positives     another piece's vertices within 0.35" in out
    more = _config(root, out_dir=str(tmp_path / "second"), steps_per_epoch=1, epochs=2,
                   init_from=str(tmp_path / "first" / "last.pt"), contact_radius=0.35)
    history = training.train(more)
    out = capsys.readouterr().out
    assert "resumed from" in out and "applies to a fresh start only" in out
    assert "started from the weights" not in out and len(history) == 2


def test_trained_weights_are_not_checked_against_chance(tmp_path, monkeypatch):
    """The first epoch's check compares the scores with what untrained weights
    read; a run from --init_from's would fail it for being trained."""
    from reassembly import training

    checked = []
    monkeypatch.setattr(training, "check_initial_losses",
                        lambda summary: checked.append(summary) or [])
    root = _dataset(tmp_path / "data")
    training.train(_config(root, out_dir=str(tmp_path / "first"), steps_per_epoch=1,
                           check_init=True))
    assert len(checked) == 1
    training.train(_config(root, out_dir=str(tmp_path / "second"), steps_per_epoch=1,
                           check_init=True, init_from=str(tmp_path / "first" / "last.pt")))
    assert len(checked) == 1


# --------------------------------------------------------------------------
# Evaluation on remeshed scenes
# --------------------------------------------------------------------------

@pytest.mark.usefixtures("thinning")
def test_an_evaluation_scores_remeshed_scenes_when_asked(tmp_path, capsys):
    from reassembly import training

    root = _dataset(tmp_path / "data")
    config = _config(root, out_dir=str(tmp_path / "run"), steps_per_epoch=1,
                     remesh=(0.5, 0.9), remesh_jitter=1e-3, remesh_gap=0.5, contact_radius=0.35)
    training.train(config)
    assert ("50%-90% of its faces kept, breaks opened up to 0.5, jitter 0.001; validation as "
            "stored") in capsys.readouterr().out
    plain = training.evaluate(_config(root, out_dir=str(tmp_path / "run")), checkpoint="last.pt",
                              split="val")
    assert "remesh" not in plain
    asked = training.evaluate(_config(root, out_dir=str(tmp_path / "run")), checkpoint="last.pt",
                              split="val", remesh=(0.5, 0.9), remesh_jitter=1e-3,
                              predictions=str(tmp_path / "predictions"))
    assert asked["remesh"] == {"low": 0.5, "high": 0.9, "jitter": 1e-3, "gap": 0.0}
    assert (tmp_path / "run" / "val_metrics_remesh0.5-0.9j0.001.json").exists()
    report = (tmp_path / "run" / "val_report_remesh0.5-0.9j0.001.txt").read_text()
    assert "remeshed          each piece on its own, 50%-90% of its faces kept" in report
    # The dumps hold the pieces as scored: remeshed, not as stored.
    import io
    import zipfile

    written = asked["evaluation"]["predictions"]["zip"]
    assert written.endswith("-val_remesh0.5-0.9j0.001.zip")
    remeshed = BreakingBadScenes(_config(root), "val", remesh=Remesh(0.5, 0.9, 1e-3))
    stored = BreakingBadScenes(_config(root), "val")
    with zipfile.ZipFile(written) as archive:
        dumps = [name for name in archive.namelist() if name.endswith(".npz")]
        assert dumps
        for name in dumps:
            arrays = np.load(io.BytesIO(archive.read(name)))
            index = remeshed.index_of(str(arrays["scene"]))
            faces = np.diff(arrays["face_offsets"]).tolist()
            assert faces == [len(m.faces) for m in remeshed.pieces(index)]
            assert sum(faces) < sum(len(m.faces) for m in stored.meshes(index))
    with pytest.raises(ValueError, match="give remesh"):
        training.evaluate(config, checkpoint="last.pt", split="val", remesh_jitter=1e-3)
    with pytest.raises(ValueError, match="give remesh"):
        training.evaluate(config, checkpoint="last.pt", split="val", remesh_gap=0.5)
    opened = training.evaluate(_config(root, out_dir=str(tmp_path / "run")), checkpoint="last.pt",
                               split="val", remesh=(0.5, 0.9), remesh_jitter=1e-3, remesh_gap=0.5)
    assert opened["remesh"] == {"low": 0.5, "high": 0.9, "jitter": 1e-3, "gap": 0.5}
    report = (tmp_path / "run" / "val_report_remesh0.5-0.9j0.001g0.5.txt").read_text()
    assert "50%-90% of its faces kept, breaks opened up to 0.5, jitter 0.001" in report


@pytest.mark.usefixtures("thinning")
def test_the_dump_script_writes_the_pieces_the_model_saw(tmp_path):
    from scripts.dump_prediction import dump

    root = _dataset(tmp_path / "data")
    config = _config(root, remesh=(0.5, 0.9), remesh_jitter=1e-3, contact_radius=0.35)
    dataset = BreakingBadScenes(config, "train")
    torch.manual_seed(0)
    model = build_model(config).eval()
    dump(dataset, 0, model, config, "cpu", seed=0, out=tmp_path / "scene.npz")
    arrays = np.load(tmp_path / "scene.npz")
    faces = np.diff(arrays["face_offsets"]).tolist()
    assert faces == [len(m.faces) for m in dataset.pieces(0)]
    assert faces != [len(m.faces) for m in dataset.meshes(0)]


def test_the_script_hands_an_evaluation_its_remeshing(monkeypatch):
    import sys

    import scripts.train as script

    seen = {}
    monkeypatch.setattr(script, "evaluate", lambda config, **kw: seen.update(kw))
    monkeypatch.setattr(sys, "argv", ["train.py", "--root_dir", "nowhere", "--evaluate",
                                      "--remesh", "0.5", "0.9", "--remesh_jitter", "0.001",
                                      "--remesh_gap", "0.03"])
    script.main()
    assert (tuple(seen["remesh"]), seen["remesh_jitter"], seen["remesh_gap"]) == (
        (0.5, 0.9), 0.001, 0.03)


def test_the_flags_carry_the_new_settings():
    import argparse

    from scripts.config_flags import add_config_arguments, config_flags, config_from_args

    parser = add_config_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--remesh", "0.5", "0.9", "--remesh_jitter", "0.001",
                              "--remesh_gap", "0.03", "--contact_radius", "0.015",
                              "--contact_band", "3", "--init_from", "runs/v7/best.pt"])
    config = config_from_args(args)
    assert (config.remesh, config.remesh_jitter, config.remesh_gap, config.contact_radius,
            config.contact_band, config.init_from) == ((0.5, 0.9), 0.001, 0.03, 0.015, 3.0,
                                                       "runs/v7/best.pt")
    again = config_from_args(parser.parse_args(config_flags(config)))
    assert again == config
    with pytest.raises(SystemExit):                       # LOW and HIGH, both
        parser.parse_args(["--remesh", "0.5"])


# --------------------------------------------------------------------------
# Can it learn breaks with no twin
# --------------------------------------------------------------------------

def _height(seed: int):
    rng = np.random.default_rng(seed)
    amplitude = rng.uniform(0.02, 0.05, 4)          # at most 0.2: no two interfaces cross
    frequency = rng.uniform(1.5, 3.5, (4, 2)) * rng.choice([-1, 1], (4, 2))
    phase = rng.uniform(0, 2 * np.pi, 4)
    return lambda x, y: sum(a * np.sin(fx * x + fy * y + p)
                            for a, (fx, fy), p in zip(amplitude, frequency, phase))


def _unshared(seed: int, pieces: int):
    """
    Slabs stacked along z and broken along smooth rough interfaces, each slab
    sampled on its own grid -- 9 x 9 and 10 x 10 by turns -- so the two sides
    of a break are the same surface with no vertex in common but the four
    corners: what two scans of a break are.
    """
    from test_assembly_placement import _slab

    thickness = [0.6 + 0.2 * ((seed + i) % 3) for i in range(pieces)]
    levels = np.concatenate([[0.0], np.cumsum(thickness)])
    interfaces = [None] + [_height(100 * seed + k) for k in range(1, pieces)] + [None]
    meshes, breaks = [], []
    for k in range(pieces):
        n = 9 if (seed + k) % 2 else 10
        x = np.linspace(-1.0, 1.0, n)
        gx, gy = np.meshgrid(x, x, indexing="ij")

        def surface(index):
            h = interfaces[index]
            z = np.full_like(gx, levels[index]) + (h(gx, gy) if h is not None else 0.0)
            return np.stack([gx.ravel(), gy.ravel(), z.ravel()], axis=1)

        meshes.append(_slab(surface(k + 1), surface(k), n))
        # `_slab` stacks the top grid over the bottom one; a grid is the break
        # where it is an interface.
        breaks.append(np.repeat([k < pieces - 1, k > 0], n * n))
    return meshes, breaks


def _unshared_scene(seed: int, pieces: int, radius: float):
    meshes, breaks = _unshared(seed, pieces)
    points = [np.asarray(m.vertices) for m in meshes]
    clusters, _ = compute_scene_correspondence(meshes)
    near = contact_vertices(points, radius)
    return build_scene(points, [np.asarray(m.faces) for m in meshes], breaks,
                       rotations=random_rotations(pieces, np.random.default_rng(100 + seed)),
                       tokens_per_scene=48, cluster=np.concatenate(clusters),
                       contact=np.concatenate(near), key=f"unshared{seed}")


def test_the_unshared_slabs_share_no_vertex_but_touch():
    meshes, breaks = _unshared(0, 3)
    points = [np.asarray(m.vertices) for m in meshes]
    assert _shared([(p, None) for p in points]) == 2 * 4 * 2   # the corners of two breaks
    assert all(m.is_watertight and m.volume > 0 for m in meshes)
    for mask, near in zip(breaks, contact_vertices(points, 0.15)):
        # The break, all but the vertices whose nearest across it is beyond
        # the radius (up to 0.17 here, on grids 0.22 and 0.25 apart), and
        # nothing else.
        assert near[mask].mean() > 0.9 and not near[~mask].any()


@pytest.mark.slow
def test_the_model_learns_breaks_that_share_no_vertex():
    """
    Six stacks of slabs whose breaks share no vertex but their corners, each
    side sampled on its own grid, fitted on the distance-based term alone: the
    rotations stage two fits from the matches go from chance to nearly all
    right. Measured: 8 of the 9 scored pieces within 5 degrees (mean 20.8,
    untrained 146.4). The same run on the coincidence term, whose only
    positives here are the four corner twins of each break, matched those
    corners (match@1 0.89) and left 7 of the 9 wrong (mean 96.8) -- what
    scanned pieces, which share no vertex at all, got from v7.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        config = Config(accumulate=1, channels=32, heads=4, lr=3e-3, batch_size=2, workers=0,
                        schedule=("intra", "intra", "cross", "intra"),
                        contact_radius=0.15, contact_band=2.0)
        scenes = [_unshared_scene(seed, 2 + seed % 2, config.contact_radius)
                  for seed in range(6)]
        loader = torch.utils.data.DataLoader(scenes, batch_size=2, shuffle=True,
                                             collate_fn=_collate_samples)
        validation = [_collate_samples(scenes[i:i + 2]) for i in range(0, 6, 2)]
        model = build_model(config)
        criterion = build_criterion(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        untrained, _, _ = run_epoch(model, validation, criterion, config, label="val",
                                    show_progress=False)
        epochs, step = 120, 0
        for epoch in range(epochs):
            summary, step, _ = run_epoch(model, loader, criterion, config,
                                         optimizer=optimizer, step=step,
                                         total_steps=3 * epochs, label="train",
                                         show_progress=False)
            if epoch == 0:
                first = summary
        trained, _, _ = run_epoch(model, validation, criterion, config, label="val",
                                  show_progress=False)
    finally:
        torch.set_num_threads(threads)

    assert untrained["acc@10deg"] < 0.5
    assert summary["embedding"] < 0.7 * first["embedding"]
    assert trained["geodesic_deg"] < 45.0 and trained["acc@10deg"] > 0.6, (
        f"after {epochs} epochs the rotations fitted from the matches are "
        f"{trained['geodesic_deg']:.1f} deg off (acc@10 {trained['acc@10deg']:.2f}; chance "
        f"{CHANCE['geodesic_deg']:.0f}): the distance-based term does not teach matching "
        f"without twins")

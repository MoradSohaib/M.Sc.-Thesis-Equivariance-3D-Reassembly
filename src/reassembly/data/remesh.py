"""
Each piece remeshed on its own: training scenes whose breaks do not share
their vertices.

Breaking Bad's pieces were cut from one mesh, so the two sides of a break
share their vertices -- the same coordinates on both pieces. Scanned pieces
never do: each is meshed on its own, and the nearest vertex across a break
lies some fraction of an edge away (median 0.0055 to 0.011 on Fractura's real
pieces and Fantastic Breaks, converted, against an edge of 0.0162). Trained only
on coincident breaks, the embedding matches a vertex to its exact twin
(:func:`reassembly.nn.losses.correspondence_loss`), and without twins it
matches at chance: Fractura's synthetic bones with each piece thinned on its
own fell from part accuracy 0.659 to 0.047 over 29 scenes.

:func:`remesh_pieces` gives the training scenes that condition. Each piece
keeps a share of its faces drawn uniformly from ``[low, high]`` -- quadric edge
collapse (:func:`reassembly.mesh.decimate.simplify`), one draw per piece, so
the two sides of a break collapse differently -- and then, with ``jitter``,
Gaussian noise of that standard deviation (world units, an object's largest
side being 1) moves every vertex, so none is left exactly where its twin was.
The two steps are also separate functions, :func:`collapse_pieces` and
:func:`jitter_pieces`, because the fracture labels have to be computed between
them: the dihedral test reads the noise as fracture. On the seven scenes below
it flags 13% to 76% of the vertices as stored, 14% to 80% after the collapse,
41% to 89% after jitter 0.0001 and 90% to 99.5% after jitter 0.001 -- and the
labels place the cross-attention tokens and bound stage two's matching.

Collapsing does not open the breaks. Remeshed, the two sides still lie within
a fraction of an edge of each other -- a median gap of 0.0015 to 0.011 to the
other piece's surface, among the vertices within 0.05 of it -- where
Fractura's real pieces, as converted, sit 0.011 to 0.029 apart, and one of
them measured at full resolution as far. :func:`open_pieces` opens them: each
piece's break vertices move along their break's normal by a smooth random
amount, inward or outward, so the two sides of a break stand apart in places
and overlap in others, as the scans' do (``gap``, below); the vertices beside
the break move with it, and no face turns over.

Measured on seven Breaking Bad scenes (8,924 to 31,792 faces, 3 to 50
pieces), three draws each, at ``low, high = 0.5, 0.9``: 18% to 46% of the
coincident vertices survive the collapse alone, and none with ``jitter``
0.001. With it, the median edge grows by a factor of 1.18 to 1.48, and a
scene takes 22 to 102 ms. A vertex of another piece within 0.015 then finds
96% to 100% of each scene's break (the vertices nearest a coincident one
before), and 66% to 96% of the vertices it finds lie on it
(:func:`reassembly.mesh.correspondence.contact_vertices`).
"""
from __future__ import annotations

from typing import List, NamedTuple, Sequence, Tuple

import numpy as np


class Remesh(NamedTuple):
    """How each piece is remeshed: the share of its faces kept is drawn from
    ``[low, high]``; with ``gap``, the breaks are opened by a mean drawn per
    scene from ``[0, gap]`` (:func:`open_pieces`); then every vertex moves by
    Gaussian noise of ``jitter``."""

    low: float
    high: float
    jitter: float = 0.0
    gap: float = 0.0


def check_remesh(low: float, high: float, jitter: float = 0.0, gap: float = 0.0) -> None:
    """Refuse a setting that would not remesh, or could not."""
    if not 0.0 < low <= high <= 1.0:
        raise ValueError(f"remesh keeps a share of each piece's faces: LOW and HIGH must "
                         f"satisfy 0 < LOW <= HIGH <= 1, got {low:g} and {high:g}")
    if jitter < 0.0:
        raise ValueError(f"remesh_jitter is a standard deviation: >= 0, got {jitter:g}")
    if gap < 0.0:
        raise ValueError(f"remesh_gap is the largest mean opening of a break: >= 0, got {gap:g}")


# How an opened break varies, measured on Fractura's 27 real scenes as the
# model got them (distance to the other pieces' surface, vertices within 0.05
# of it): the gap of two vertices of a piece correlates 0.96 at 0-0.01 apart,
# 0.84 at 0.01-0.02, 0.51 at 0.02-0.04 and 0.02 at 0.04-0.08 (medians over the
# scenes), which a Gaussian field of length scale 0.026 reproduces (0.85 at
# 0.015, 0.51 at 0.03, 0.07 at 0.06). Each side moves by its own field, with a
# standard deviation equal to its mean. On seven Breaking Bad scenes, five
# draws each at --remesh 0.5 0.9 --remesh_jitter 0.001 --remesh_gap 0.03, the
# 18 draws opened by 0.01 or more on six of them give 26% to 57% of the contact
# within 0.005 (median 34%), a first quartile of 0.0047 to 0.0130 (median
# 0.0088) and a median of 0.0117 to 0.0226 (median 0.0190); the scans, 20% to
# 55% (39%), 0.0038 to 0.0175 (0.0081) and 0.011 to 0.029 (0.0184). A spread of
# 1.8 gave 42% to 57% (52%) and 0.0034 to 0.0069 (0.0045). The seventh, a spoon
# in 50 pieces too thin for the step, opens less: a median of 0.0052 to 0.0075.
GAP_LENGTH = 0.026
GAP_SPREAD = 1.0
# Random Fourier features of that field.
GAP_TERMS = 32
# Moved along its own normal, each vertex of a rough break goes its own way: in
# the draws above opened by 0.01 or more (23, the spoon's five included), that
# turned over 8% to 25% of the break's faces, and 14% to 49% of the faces
# beside it, whose other corners stay. So the direction is the break's normal
# averaged over GAP_LENGTH (0% to 5% of the break's faces turned), and the
# vertices beside the break move with it: within GAP_REACH times a break
# vertex's step of it, each moves as its nearest break vertex does, by a weight
# falling from 1 to 0 (a smoothstep, whose steepest slope, 1.5, squeezes the
# faces between by at most a half; 0.5% to 9% of the faces beside the break
# turned). Wherever a face still turns over -- 0.03% to 2% of all faces on the
# six scenes, 4% to 9% on the spoon -- its corners move half as far, again
# until none does, which keeps 55% to 96% of the step on the six and 29% to 54%
# on the spoon.
GAP_REACH = 3.0


def collapse_pieces(vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray],
                    rng: np.random.Generator, spec: Remesh
                    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    ``(vertices, faces)`` per piece, each collapsed on its own, without the
    jitter: the surface the fracture labels are computed on. The shares are
    drawn first, one per piece, so a piece's share does not depend on the
    pieces before it. A piece of :data:`~reassembly.mesh.decimate.MIN_FACES`
    faces or fewer keeps its faces, and so does one the collapse would leave
    empty. The collapse is tidied as the converter tidies it -- vertices
    merged, the faces that collapse and repeated faces removed
    (:func:`~reassembly.mesh.clean.clean_piece`), every component kept -- and
    keeps the winding the collapse leaves, the piece's own.
    """
    from ..mesh.clean import clean_piece
    from ..mesh.decimate import MIN_FACES, simplify

    shares = rng.uniform(spec.low, spec.high, size=len(vertices))
    out_vertices, out_faces = [], []
    for v, f, share in zip(vertices, faces, shares):
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)
        target = max(MIN_FACES, int(round(len(f) * share)))
        if target < len(f):
            piece = clean_piece(*simplify(v, f, target), drop_debris=False)
            if len(piece.faces):
                # clean_piece turns a closed piece wound inward outward: undone,
                # so remeshing changes the tessellation and nothing else.
                v, f = piece.vertices, (np.ascontiguousarray(piece.faces[:, ::-1])
                                        if piece.report.flipped else piece.faces)
        out_vertices.append(v)
        out_faces.append(f)
    return out_vertices, out_faces


def jitter_pieces(vertices: Sequence[np.ndarray], rng: np.random.Generator,
                  jitter: float) -> List[np.ndarray]:
    """Every vertex moved by Gaussian noise of standard deviation ``jitter``,
    piece by piece in order; 0 leaves them where they are."""
    vertices = [np.asarray(v, dtype=np.float64) for v in vertices]
    if not jitter:
        return vertices
    return [v + rng.normal(0.0, jitter, v.shape) for v in vertices]


def break_masks(stored: Sequence[np.ndarray], vertices: Sequence[np.ndarray]
                ) -> List[np.ndarray]:
    """
    Per piece of ``vertices`` (the collapsed pieces), which vertices are on a
    break: those whose nearest vertex of the ``stored`` piece coincides with a
    vertex of another stored piece -- Breaking Bad's break, exact in the stored
    pieces (:func:`~reassembly.mesh.fracture.fracture_vertex_masks`), carried
    over to the collapse.
    """
    from scipy.spatial import cKDTree

    from ..mesh.fracture import fracture_vertex_masks

    stored = [np.asarray(v, dtype=np.float64) for v in stored]
    twins = fracture_vertex_masks(stored)
    return [twin[cKDTree(s).query(np.asarray(v, dtype=np.float64))[1]] if len(s) else
            np.zeros(len(v), dtype=bool) for s, v, twin in zip(stored, vertices, twins)]


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Each face's normal, its length twice the face's area."""
    return np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                    vertices[faces[:, 2]] - vertices[faces[:, 0]])


def _break_directions(vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Per break vertex (``mask``), the unit normal of the break about it: each
    vertex's normal area-weighted over its break faces (all three corners on
    the break), over all its faces where it has none, then averaged over the
    break with a Gaussian of width :data:`GAP_LENGTH` (to three widths).
    """
    from scipy.spatial import cKDTree

    normal = _face_normals(vertices, faces)
    on_break = mask[faces].all(axis=1)
    near, every = np.zeros_like(vertices), np.zeros_like(vertices)
    for k in range(3):
        np.add.at(near, faces[on_break, k], normal[on_break])
        np.add.at(every, faces[:, k], normal)
    own = np.where((np.linalg.norm(near, axis=1) > 0)[:, None], near, every)[mask]
    tree = cKDTree(vertices[mask])
    pairs = tree.sparse_distance_matrix(tree, 3.0 * GAP_LENGTH, output_type="ndarray")
    pairs = pairs[pairs["i"] != pairs["j"]]               # each vertex counted once, below
    weight = np.exp(-0.5 * (pairs["v"] / GAP_LENGTH) ** 2)
    average = own.copy()
    for k in range(3):
        average[:, k] += np.bincount(pairs["i"], weight * own[pairs["j"], k], minlength=len(own))
    return average / np.maximum(np.linalg.norm(average, axis=1, keepdims=True), 1e-30)


def _unturned(vertices: np.ndarray, faces: np.ndarray, step: np.ndarray) -> np.ndarray:
    """``step`` cut back wherever it would turn a face over -- its normal more
    than 90 degrees from where it was: the corners of such a face move half as
    far, again until no face turns; after eight halvings, not at all. A face
    of no area has no side to turn to and is left out."""
    before = _face_normals(vertices, faces)
    checked = np.linalg.norm(before, axis=1) > 0
    scale = np.ones(len(vertices))
    while True:
        after = _face_normals(vertices + scale[:, None] * step, faces)
        turned = checked & (np.einsum("ij,ij->i", before, after) <= 0.0)
        if not turned.any():
            return scale[:, None] * step
        corners = np.unique(faces[turned])
        scale[corners] = np.where(scale[corners] > 0.5 ** 8, 0.5 * scale[corners], 0.0)


def open_pieces(vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray],
                breaks: Sequence[np.ndarray], rng: np.random.Generator,
                gap: float) -> List[np.ndarray]:
    """
    Every piece's break vertices (``breaks``) moved along their break's normal
    (:func:`_break_directions`) by ``s(x) = m (1 + GAP_SPREAD g(x))`` -- inward
    where positive, outward, into the other piece, where negative -- with
    ``g`` a smooth unit-variance random field of length scale
    :data:`GAP_LENGTH`, one per piece, and ``m`` half of a mean opening drawn
    once from ``[0, gap]``: the two sides of a break stand ``2m`` apart on
    average, touch or overlap in places, and a scene can come out barely
    opened. A vertex within ``GAP_REACH |s|`` of the break moves as its
    nearest break vertex does, by a smoothstep weight from 1 at the break to 0
    there, and no face turns over (:func:`_unturned`). Draws the same numbers
    whatever the masks hold; 0 leaves the pieces where they are.
    """
    from scipy.spatial import cKDTree

    vertices = [np.asarray(v, dtype=np.float64) for v in vertices]
    if not gap:
        return vertices
    half = 0.5 * gap * rng.uniform()
    out = []
    for v, f, mask in zip(vertices, faces, breaks):
        omega = rng.normal(0.0, 1.0 / GAP_LENGTH, (GAP_TERMS, 3))
        phase = rng.uniform(0.0, 2.0 * np.pi, GAP_TERMS)
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            f = np.asarray(f, dtype=np.int64)
            field = np.sqrt(2.0 / GAP_TERMS) * np.cos(v[mask] @ omega.T + phase).sum(axis=1)
            s = half * (1.0 + GAP_SPREAD * field)
            step = -s[:, None] * _break_directions(v, f, mask)
            distance, nearest = cKDTree(v[mask]).query(v)
            u = np.clip(distance / np.maximum(GAP_REACH * np.abs(s[nearest]), 1e-30), 0.0, 1.0)
            weight = 1.0 - u * u * (3.0 - 2.0 * u)
            weight[mask] = 1.0
            spread = weight[:, None] * step[nearest]
            spread[mask] = step
            v = v + _unturned(v, f, spread)
        out.append(v)
    return out


def remesh_pieces(vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray],
                  rng: np.random.Generator, spec: Remesh
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    ``(vertices, faces)`` per piece, each remeshed on its own (module
    docstring): :func:`collapse_pieces`; with ``spec.gap``, :func:`open_pieces`
    on the breaks the input pieces share (:func:`break_masks`); then
    :func:`jitter_pieces` with ``spec.jitter`` -- from the same generator. The
    opening and the jitter move the vertices, never the faces.
    """
    collapsed, faces = collapse_pieces(vertices, faces, rng, spec)
    if spec.gap:
        collapsed = open_pieces(collapsed, faces, break_masks(vertices, collapsed), rng, spec.gap)
    return jitter_pieces(collapsed, rng, spec.jitter), faces


__all__ = ["GAP_LENGTH", "GAP_REACH", "GAP_SPREAD", "GAP_TERMS", "Remesh", "break_masks",
           "check_remesh", "collapse_pieces", "jitter_pieces", "open_pieces", "remesh_pieces"]

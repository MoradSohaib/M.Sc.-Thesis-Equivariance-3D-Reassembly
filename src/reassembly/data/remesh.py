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
side being 1) moves every break vertex (:func:`break_masks`), so none is left
exactly where its twin was.
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

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


class Remesh(NamedTuple):
    """
    How each piece is remeshed, and how its breaks are then made to look
    scanned.

    Remeshing: the share of its faces kept is drawn from ``[low, high]``
    (quadric collapse, :func:`collapse_pieces`) -- or, with ``edge``, each
    piece is remeshed isotropically to an edge length drawn from that range
    (:func:`isotropic_pieces`), in place of the collapse.

    The breaks, in this order: with ``gap``, opened by a mean drawn per scene
    from ``[0, gap]`` (:func:`open_pieces`); with ``smooth``, worn by up to
    that many smoothing steps (:func:`smooth_pieces`); with ``deform``, bent
    by a slow field of up to that amplitude (:func:`deform_pieces`); then, after
    the fracture labels are read, with ``noise``, rippled along the normal by
    a spatially correlated field (:func:`noise_pieces`); and every break vertex
    moved by Gaussian noise of ``jitter``.

    ``remesh_probability`` and ``augment_probability`` are the shares of the
    scenes remeshed and perturbed (one number per scene and epoch decides both:
    a scene is perturbed only if it is also remeshed, unless the perturbing
    share is the larger; in training the sampler deals those numbers so every
    optimizer step holds its exact shares); the rest are the scenes as stored.
    """

    low: float
    high: float
    jitter: float = 0.0
    gap: float = 0.0
    edge: Optional[Tuple[float, float]] = None
    smooth: int = 0
    noise: float = 0.0
    deform: float = 0.0
    remesh_probability: float = 1.0
    augment_probability: float = 1.0

    @property
    def perturbs(self) -> bool:
        """Whether anything moves the vertices once the pieces are remeshed."""
        return bool(self.jitter or self.gap or self.smooth or self.noise or self.deform)


def check_remesh(low: float, high: float, jitter: float = 0.0, gap: float = 0.0) -> None:
    """Refuse a setting that would not remesh, or could not."""
    if not 0.0 < low <= high <= 1.0:
        raise ValueError(f"remesh keeps a share of each piece's faces: LOW and HIGH must "
                         f"satisfy 0 < LOW <= HIGH <= 1, got {low:g} and {high:g}")
    if jitter < 0.0:
        raise ValueError(f"remesh_jitter is a standard deviation: >= 0, got {jitter:g}")
    if gap < 0.0:
        raise ValueError(f"remesh_gap is the largest mean opening of a break: >= 0, got {gap:g}")


def check_augment(edge: Optional[Sequence[float]] = None, smooth: int = 0, noise: float = 0.0,
                  deform: float = 0.0, remesh_probability: float = 1.0,
                  augment_probability: float = 1.0) -> None:
    """Refuse the settings beyond :func:`check_remesh` that could not work."""
    if edge is not None:
        if len(edge) != 2 or not 0.0 < edge[0] <= edge[1]:
            raise ValueError(f"remesh_edge is two edge lengths, 0 < LOW <= HIGH, in world "
                             f"units, got {edge!r}")
    if smooth < 0:
        raise ValueError(f"remesh_smooth is a number of smoothing steps: >= 0, got {smooth}")
    if noise < 0.0:
        raise ValueError(f"remesh_noise is an amplitude: >= 0, got {noise:g}")
    if deform < 0.0:
        raise ValueError(f"remesh_deform is an amplitude: >= 0, got {deform:g}")
    for name, value in (("remesh_probability", remesh_probability),
                        ("augment_probability", augment_probability)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} is a share of the scenes: in [0, 1], got {value:g}")


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


def jitter_pieces(vertices: Sequence[np.ndarray], breaks: Sequence[np.ndarray],
                  rng: np.random.Generator, jitter: float) -> List[np.ndarray]:
    """Every piece's break vertices (``breaks``) moved by Gaussian noise of
    standard deviation ``jitter``, piece by piece in order; the other vertices
    stay where they are. Draws the same numbers whatever the masks hold; 0
    leaves the pieces where they are."""
    vertices = [np.asarray(v, dtype=np.float64) for v in vertices]
    if not jitter:
        return vertices
    out = []
    for v, mask in zip(vertices, breaks):
        noise = rng.normal(0.0, jitter, v.shape)
        out.append(v + np.asarray(mask, dtype=bool)[:, None] * noise)
    return out


def carry_masks(stored: Sequence[np.ndarray], masks: Sequence[np.ndarray],
                vertices: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    Per piece, a per-vertex mask of the ``stored`` piece carried over to the
    same piece remeshed (``vertices``, before anything moved them): each
    vertex takes the value of the stored vertex nearest to it.
    """
    from scipy.spatial import cKDTree

    out = []
    for s, mask, v in zip(stored, masks, vertices):
        s = np.asarray(s, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        out.append(np.asarray(mask, dtype=bool)[cKDTree(s).query(v)[1]] if len(s) else
                   np.zeros(len(v), dtype=bool))
    return out


def break_masks(stored: Sequence[np.ndarray], vertices: Sequence[np.ndarray]
                ) -> List[np.ndarray]:
    """
    Per piece of ``vertices`` (the collapsed pieces), which vertices are on a
    break: those whose nearest vertex of the ``stored`` piece coincides with a
    vertex of another stored piece -- Breaking Bad's break, exact in the stored
    pieces (:func:`~reassembly.mesh.fracture.fracture_vertex_masks`), carried
    over to the collapse (:func:`carry_masks`).
    """
    from ..mesh.fracture import fracture_vertex_masks

    stored = [np.asarray(v, dtype=np.float64) for v in stored]
    return carry_masks(stored, fracture_vertex_masks(stored), vertices)


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


# Isotropic remeshing is pymeshlab's explicit remesher (Botsch and Kobbelt's
# split, collapse, flip, relax and reproject), three passes. On a 20k-face
# ellipsoid remeshed to an edge of 0.03, three passes took 0.47 s against
# 0.74 s for five, with the same spread of edges: 5th to 95th percentile 0.73
# to 1.25 of the target against 0.76 to 1.24, median 1.03 either way.
ISOTROPIC_ITERATIONS = 3
# The remesher keeps creases sharper than its feature angle as they are. Kept,
# a rough break's creases are kept on BOTH sides alike, and so are its shared
# vertices: on the tests' fractured solid, remeshed to 0.12, 1,350 vertices
# still coincided at the default 30 degrees and 686 at 90 (1,152 before), but
# 18 with none kept, at 180 -- which also rounds the rim a little, as a scan's.
ISOTROPIC_FEATURE_ANGLE = 180.0
# Without the remesher's check of the distance to the input surface a Breaking
# Bad scene (15 pieces, 34k faces) remeshed in 0.50 s against 1.38 s, at the
# same edges (median 1.03 of the target, 0.1% slivers); what the check buys is
# the sharpest corners, rounded by up to 0.8 of an edge at the 99th percentile
# of the input's vertices instead of 0.4 -- as a scan rounds them. So training
# remeshes without it, and the converter (once per scan) with it.
#
# A remesh is refused, and the piece handled otherwise, when it comes out
# worse than it went in: more than ISOTROPIC_SLIVERS of its faces slivers
# (longest edge over 5 times its altitude), a median edge outside
# ISOTROPIC_EDGE_RANGE of the target, or non-manifold edges grown by more than
# ISOTROPIC_NONMANIFOLD of its edges. Fantastic Breaks' restoration of 00002 --
# 91% slivers, 341 edges shared by four or six faces, an Euler characteristic
# of -861, a boolean cut's mesh -- came out of every setting tried with 36-49%
# slivers and a median edge of 0.28-0.45 of the target; its broken piece, a
# manifold scan, with 0.2% and 1.02.
ISOTROPIC_SLIVERS = 0.2
ISOTROPIC_EDGE_RANGE = (0.6, 1.6)
ISOTROPIC_NONMANIFOLD = 0.002


def _pymeshlab():
    """pymeshlab with its remeshing filters loaded, or an error saying what is missing."""
    try:
        import pymeshlab
    except ImportError as error:
        raise ImportError("isotropic remeshing (--remesh_edge, --isotropic) needs pymeshlab: "
                          "pip install pymeshlab") from error
    if not hasattr(pymeshlab.MeshSet(verbose=False), "meshing_isotropic_explicit_remeshing"):
        # Its meshing plugin links the OpenGL library; without it pymeshlab
        # imports, prints that the plugin did not load, and has no filter.
        raise ImportError("pymeshlab is installed but its remeshing filters did not load: on "
                          "Linux they need the OpenGL library, apt-get install -y libopengl0")
    return pymeshlab


def _sliver_share(vertices: np.ndarray, faces: np.ndarray) -> float:
    """The share of faces whose longest edge is more than 5 times its altitude."""
    corners = vertices[faces]
    sides = np.stack([np.linalg.norm(corners[:, (k + 1) % 3] - corners[:, k], axis=1)
                      for k in range(3)], axis=1)
    doubled = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                      corners[:, 2] - corners[:, 0]), axis=1)
    longest = sides.max(axis=1)
    return float(np.mean(5.0 * doubled < longest * longest)) if len(faces) else 0.0


def try_isotropic(vertices: np.ndarray, faces: np.ndarray, edge: float,
                  iterations: int = ISOTROPIC_ITERATIONS, *, fidelity: bool = False):
    """
    One piece remeshed isotropically -- triangles close to equilateral, edges
    close to ``edge`` everywhere, in the piece's own units -- then tidied as
    :func:`collapse_pieces` tidies a collapse, keeping the piece's winding.
    ``(vertices, faces)``, or ``None`` when the remesh failed: fewer than
    :data:`~reassembly.mesh.decimate.MIN_FACES` faces, or worse than the input
    (:data:`ISOTROPIC_SLIVERS`). ``fidelity`` keeps the remesher's check of the
    distance to the input surface (:data:`ISOTROPIC_FEATURE_ANGLE`'s note).
    """
    from ..mesh.clean import clean_piece, edge_counts
    from ..mesh.decimate import MIN_FACES, edge_lengths

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) < MIN_FACES:
        return None
    pymeshlab = _pymeshlab()
    meshes = pymeshlab.MeshSet(verbose=False)
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    meshes.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(float(edge)),
                                                iterations=int(iterations),
                                                featuredeg=ISOTROPIC_FEATURE_ANGLE,
                                                checksurfdist=bool(fidelity))
    out = meshes.current_mesh()
    remeshed = np.asarray(out.face_matrix(), dtype=np.int64)
    if len(remeshed) < MIN_FACES:
        return None
    piece = clean_piece(np.asarray(out.vertex_matrix(), dtype=np.float64), remeshed,
                        drop_debris=False)
    if len(piece.faces) < MIN_FACES:
        return None
    ratio = float(np.median(edge_lengths(piece.vertices, piece.faces))) / float(edge)
    grown = edge_counts(piece.faces)[1] - edge_counts(faces)[1]
    if (_sliver_share(piece.vertices, piece.faces) > ISOTROPIC_SLIVERS
            or not ISOTROPIC_EDGE_RANGE[0] <= ratio <= ISOTROPIC_EDGE_RANGE[1]
            or grown > ISOTROPIC_NONMANIFOLD * 1.5 * len(piece.faces)):
        return None
    # As in collapse_pieces: the tidy turns a piece wound inward outward.
    return piece.vertices, (np.ascontiguousarray(piece.faces[:, ::-1]) if piece.report.flipped
                            else piece.faces)


def remesh_isotropic(vertices: np.ndarray, faces: np.ndarray, edge: float,
                     iterations: int = ISOTROPIC_ITERATIONS, *, fidelity: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    One piece remeshed isotropically (:func:`try_isotropic`); where that
    fails, collapsed instead to the faces an even mesh of edge ``edge`` would
    have -- its tessellation still its own -- or, already that coarse, kept.

    Why: an intra-fragment layer reaches one ring of the mesh, so what the
    network sees around a vertex, in world units, is the number of layers
    times the edge length. Breaking Bad's pieces are simulated, with long thin
    triangles, and scanned pieces are meshed evenly; remeshed both to one edge
    length, they are seen at one scale and in one triangulation.
    """
    from ..mesh.clean import clean_piece
    from ..mesh.decimate import MIN_FACES, simplify, surface_area

    result = try_isotropic(vertices, faces, edge, iterations, fidelity=fidelity)
    if result is not None:
        return result
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    # Faces of an even mesh of that edge: the area over an equilateral triangle's.
    target = int(round(surface_area(vertices, faces) / (np.sqrt(3.0) / 4.0 * edge * edge)))
    if len(faces) < MIN_FACES or target >= len(faces):
        return vertices, faces
    piece = clean_piece(*simplify(vertices, faces, max(MIN_FACES, target)), drop_debris=False)
    if not len(piece.faces):
        return vertices, faces
    return piece.vertices, (np.ascontiguousarray(piece.faces[:, ::-1]) if piece.report.flipped
                            else piece.faces)


def thin_keeping_topology(vertices: np.ndarray, faces: np.ndarray,
                          edge: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    A dense piece collapsed to the faces an even mesh of edge ``edge`` would
    have, by pymeshlab's quadric edge collapse with its topology kept -- the
    converter's step before remeshing a full-resolution scan. The faster
    collapse :func:`~reassembly.mesh.decimate.simplify` uses does not keep it:
    on Fractura's pig_bone_1 it left 12 to 173 edges shared by more than two
    faces in each of the four pieces, which the remesher then carried on and
    opened into holes (360 boundary edges). Kept, none, and the remesh came out
    as closed as the scan.
    """
    from ..mesh.decimate import MIN_FACES, surface_area

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    target = int(round(surface_area(vertices, faces) / (np.sqrt(3.0) / 4.0 * edge * edge)))
    if target >= len(faces):
        return vertices, faces
    pymeshlab = _pymeshlab()
    meshes = pymeshlab.MeshSet(verbose=False)
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    meshes.meshing_decimation_quadric_edge_collapse(targetfacenum=max(target, MIN_FACES),
                                                    preservetopology=True,
                                                    preserveboundary=True, qualitythr=0.3)
    out = meshes.current_mesh()
    thinned = np.asarray(out.face_matrix(), dtype=np.int64)
    if len(thinned) < MIN_FACES:           # nothing usable left: kept as it came
        return vertices, faces
    return np.asarray(out.vertex_matrix(), dtype=np.float64), thinned


def isotropic_pieces(vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray],
                     rng: np.random.Generator, low: float, high: float,
                     iterations: int = ISOTROPIC_ITERATIONS
                     ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    ``(vertices, faces)`` per piece, each remeshed isotropically on its own
    (:func:`remesh_isotropic`) to an edge length drawn per piece from
    ``[low, high]``, in world units -- drawn first, one per piece, so a piece's
    edge does not depend on the pieces before it. The two sides of a break
    come out tessellated independently, at the edge length the scans are
    converted to (``scripts/convert_external.py --isotropic``).
    """
    edges = rng.uniform(low, high, size=len(vertices))
    out_vertices, out_faces = [], []
    for v, f, edge in zip(vertices, faces, edges):
        v, f = remesh_isotropic(v, f, edge, iterations)
        out_vertices.append(v)
        out_faces.append(f)
    return out_vertices, out_faces


def _neighbour_mean(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Each vertex's mean neighbour along the mesh's edges; itself when it has none."""
    from ..mesh.topology import unique_edges

    edges = unique_edges(faces, len(vertices))
    source = np.concatenate([edges[:, 0], edges[:, 1]])
    target = np.concatenate([edges[:, 1], edges[:, 0]])
    total = np.zeros_like(vertices)
    np.add.at(total, target, vertices[source])
    count = np.bincount(target, minlength=len(vertices)).astype(np.float64)
    return np.where(count[:, None] > 0, total / np.maximum(count, 1.0)[:, None], vertices)


# Wear: each smoothing step moves a break vertex this share of the way to the
# mean of its neighbours, and a vertex one edge from the break (the intact
# side of the rim) half as far, so the rim rounds off instead of standing as a
# ridge between a smoothed break and a sharp intact edge. A step is capped at
# SMOOTH_CAP of the piece's median edge: the mean of a vertex's neighbours lies
# far along a long thin triangle -- Breaking Bad's tessellation has them -- and
# uncapped, a rim vertex next to one would be dragged across the surface.
SMOOTH_STEP = 0.5
SMOOTH_BESIDE = 0.5
SMOOTH_CAP = 0.25


def smooth_pieces(vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray],
                  breaks: Sequence[np.ndarray], rng: np.random.Generator,
                  steps: int) -> List[np.ndarray]:
    """
    Every piece's break worn: ``k`` steps of Laplacian smoothing, ``k`` drawn
    per piece from ``0..steps``, on the break vertices (``breaks``) and the
    vertices one edge from them (:data:`SMOOTH_STEP`, :data:`SMOOTH_BESIDE`).
    The sharp rim where a simulated break meets the intact surface rounds
    off, and the break's finest roughness goes, as on a scanned break -- whose
    rim the scanner and handling have rounded, and whose finest detail the
    scan does not resolve. Each step is capped at :data:`SMOOTH_CAP` of the
    piece's median edge, and no face ends up turned over from where it
    started (:func:`_unturned`). Draws the same numbers whatever the masks
    hold; 0 leaves the pieces as they are.
    """
    from ..mesh.decimate import edge_lengths
    from ..mesh.topology import unique_edges

    vertices = [np.asarray(v, dtype=np.float64) for v in vertices]
    if not steps:
        return vertices
    counts = rng.integers(0, int(steps) + 1, size=len(vertices))
    out = []
    for start, f, mask, count in zip(vertices, faces, breaks, counts):
        mask = np.asarray(mask, dtype=bool)
        f = np.asarray(f, dtype=np.int64)
        v = start
        if count and mask.any() and len(f):
            edges = unique_edges(f, len(v))
            beside = np.zeros(len(v), dtype=bool)
            beside[edges[mask[edges[:, 0]], 1]] = True
            beside[edges[mask[edges[:, 1]], 0]] = True
            weight = np.where(mask, 1.0, np.where(beside, SMOOTH_BESIDE, 0.0))
            cap = SMOOTH_CAP * float(np.median(edge_lengths(start, f)))
            for _ in range(int(count)):
                step = SMOOTH_STEP * weight[:, None] * (_neighbour_mean(v, f) - v)
                length = np.linalg.norm(step, axis=1, keepdims=True)
                v = v + step * np.minimum(1.0, cap / np.maximum(length, 1e-30))
            # Turned over is judged against the piece as it came, not step by
            # step: small turns add up.
            v = start + _unturned(start, f, v - start)
        out.append(v)
    return out


# Scanner noise: a ripple along the normal, correlated between neighbouring
# vertices rather than independent per vertex as jitter_pieces' is. A Gaussian
# field of length scale NOISE_LENGTH -- about 1.4 of Breaking Bad's median
# edge, 0.0145 of an object -- as NOISE_TERMS random Fourier features.
NOISE_LENGTH = 0.02
NOISE_TERMS = 32


def noise_pieces(vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray],
                 rng: np.random.Generator, amplitude: float) -> List[np.ndarray]:
    """
    Every vertex moved along its normal by ``sigma g(x)``: ``g`` a smooth
    unit-variance random field of length scale :data:`NOISE_LENGTH`, one per
    piece, and ``sigma`` drawn once per scene from ``[0, amplitude]``, in world
    units. No face turns over (:func:`_unturned`). Draws the same numbers
    whatever the pieces hold; 0 leaves them where they are.
    """
    from ..mesh.topology import vertex_normals

    vertices = [np.asarray(v, dtype=np.float64) for v in vertices]
    if not amplitude:
        return vertices
    sigma = amplitude * rng.uniform()
    out = []
    for v, f in zip(vertices, faces):
        omega = rng.normal(0.0, 1.0 / NOISE_LENGTH, (NOISE_TERMS, 3))
        phase = rng.uniform(0.0, 2.0 * np.pi, NOISE_TERMS)
        f = np.asarray(f, dtype=np.int64)
        if len(v) and len(f):
            field = np.sqrt(2.0 / NOISE_TERMS) * np.cos(v @ omega.T + phase).sum(axis=1)
            v = v + _unturned(v, f, sigma * field[:, None] * vertex_normals(v, f))
        out.append(v)
    return out


# A slow bend: each axis of the displacement a random Fourier field of length
# scale DEFORM_LENGTH -- about a third of an object, whose largest side is 1 --
# so the two sides of a break, each bent its own way, fit nowhere exactly.
DEFORM_LENGTH = 0.3
DEFORM_TERMS = 16


def deform_pieces(vertices: Sequence[np.ndarray], rng: np.random.Generator,
                  amplitude: float) -> List[np.ndarray]:
    """
    Every piece bent by its own slow displacement field (:data:`DEFORM_LENGTH`)
    of standard deviation drawn once per scene from ``[0, amplitude]``, in
    world units -- as two scans registered each with its own small error do
    not fit exactly. Keep it well under stage two's RANSAC tolerance, 0.05 of
    the largest piece's radius: the fit is rigid. Over a field this slow no
    face can turn over at any amplitude worth using. 0 leaves the pieces
    where they are.
    """
    vertices = [np.asarray(v, dtype=np.float64) for v in vertices]
    if not amplitude:
        return vertices
    sigma = amplitude * rng.uniform()
    out = []
    for v in vertices:
        omega = rng.normal(0.0, 1.0 / DEFORM_LENGTH, (3, DEFORM_TERMS, 3))
        phase = rng.uniform(0.0, 2.0 * np.pi, (3, DEFORM_TERMS))
        if len(v):
            displacement = np.stack(
                [np.sqrt(2.0 / DEFORM_TERMS) * np.cos(v @ omega[axis].T + phase[axis]).sum(axis=1)
                 for axis in range(3)], axis=1)
            v = v + sigma * displacement
        out.append(v)
    return out


def remesh_scene(vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray],
                 rng: np.random.Generator, spec: Remesh, *, remesh: bool = True,
                 perturb: bool = True):
    """
    ``(closed, surface, moved, faces)``, every piece through ``spec`` (see
    :class:`Remesh`), all three sharing ``faces``:

    * ``closed`` -- remeshed (:func:`collapse_pieces`, or with ``spec.edge``
      :func:`isotropic_pieces`), nothing moved: where the positives of the
      distance-based term are found;
    * ``surface`` -- then opened, smoothed and bent: what the fracture labels
      are computed on;
    * ``moved`` -- then the correlated noise and the jitter: what the model
      reads. The dihedral test reads noise as fracture (module docstring), so
      the labels are computed before it.

    ``remesh=False`` keeps the stored tessellation, ``perturb=False`` moves
    nothing. One generator, drawn in that order -- with only ``gap`` and
    ``jitter`` set, the numbers :func:`remesh_pieces` has always drawn.
    """
    vertices = [np.asarray(v, dtype=np.float64) for v in vertices]
    faces = [np.asarray(f, dtype=np.int64) for f in faces]
    if remesh:
        if spec.edge is not None:
            closed, faces = isotropic_pieces(vertices, faces, rng, *spec.edge)
        else:
            closed, faces = collapse_pieces(vertices, faces, rng, spec)
    else:
        closed = vertices
    if not perturb:
        return closed, closed, closed, faces
    breaks = None
    surface = closed
    if spec.gap or spec.smooth or spec.jitter:
        breaks = break_masks(vertices, closed)
    if spec.gap:
        surface = open_pieces(surface, faces, breaks, rng, spec.gap)
    if spec.smooth:
        surface = smooth_pieces(surface, faces, breaks, rng, spec.smooth)
    if spec.deform:
        surface = deform_pieces(surface, rng, spec.deform)
    moved = noise_pieces(surface, faces, rng, spec.noise) if spec.noise else surface
    return closed, surface, jitter_pieces(moved, breaks, rng, spec.jitter), faces


def remesh_pieces(vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray],
                  rng: np.random.Generator, spec: Remesh
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    ``(vertices, faces)`` per piece, each remeshed on its own (module
    docstring): :func:`collapse_pieces`; with ``spec.gap``, :func:`open_pieces`
    on the breaks the input pieces share (:func:`break_masks`); then
    :func:`jitter_pieces` with ``spec.jitter`` on the same breaks -- from the
    same generator. The opening and the jitter move the vertices, never the
    faces. The rest of ``spec`` as :func:`remesh_scene` applies it.
    """
    _closed, _surface, moved, faces = remesh_scene(vertices, faces, rng, spec)
    return moved, faces


__all__ = ["DEFORM_LENGTH", "GAP_LENGTH", "GAP_REACH", "GAP_SPREAD", "GAP_TERMS",
           "ISOTROPIC_ITERATIONS", "NOISE_LENGTH", "Remesh", "break_masks", "carry_masks",
           "check_augment", "check_remesh", "collapse_pieces", "deform_pieces",
           "isotropic_pieces", "jitter_pieces", "noise_pieces", "open_pieces",
           "remesh_isotropic", "remesh_pieces", "remesh_scene", "smooth_pieces",
           "thin_keeping_topology", "try_isotropic"]

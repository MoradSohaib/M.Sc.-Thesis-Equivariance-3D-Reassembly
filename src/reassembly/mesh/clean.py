"""
Fragment meshes from other datasets, made like the ones the model was trained
on: each piece one closed surface.

Breaking Bad's fragments reach the network clean because the reader makes them
so -- it merges coincident vertices and cancels the interior walls of the cells
a piece was cut from (:class:`reassembly.data.scene.SceneReader`). Meshes from
elsewhere arrive without that, and each of these was found in Fractura:

* **Vertices repeated per face.** The real bones (``bone_real.hdf5``) store
  2.65 to 2.91 vertices per face, the copies slightly apart: merged with
  trimesh's default (8 decimals) a piece still reached the model as loose
  triangles -- 518,631 vertices in 168,427 separate patches, where merging
  within 1e-6 of its size gives 90,179 vertices on one closed surface. With no
  shared vertices there are no mesh edges between faces to pass messages
  along, and a vertex normal is just its one face's normal.
* **Doubled interior walls.** Synthetic pieces (simulated like Breaking Bad's)
  hold the faces between two cells of one piece twice, facing opposite ways:
  1,142 and 2,270 such pairs in the two pieces of one scene. They are
  cancelled exactly by the rule the Breaking Bad reader already applies
  (:func:`~reassembly.mesh.repair.resolve_duplicated_faces`); after it each
  of the 20 synthetic pieces checked (three scenes) was one surface with no
  edge shared by more than two faces, and all but one closed -- that one has 3
  open edges as stored.
* **Debris.** Every real bone piece also carries Blender's default cube -- a
  separate 2 x 2 x 2 box, 12 triangles, centred on the world origin, where all
  the pieces' cubes coincide in the assembled pose.
* **A piece stored twice.** ``pig_bone_15``'s two pieces are one mesh, and
  ``vert8``'s second piece is also a component of its first
  (:func:`identical_pieces`, :func:`copies_of_pieces`).

Not a ray-cast "shell" (:mod:`reassembly.mesh.shell`): visibility from outside
would also delete real surface no outside ray reaches -- the inside of a
narrow bottle's neck, a deep crack -- and the inner faces found here are
exactly the cancelled duplicates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..arrays import compact_indices
from .repair import resolve_duplicated_faces

# Vertices closer than this share of the piece's bounding-box diagonal are one
# vertex. The real bones' copies are closer than float32 resolves (they
# coincide in a float32 dump) but not within trimesh's 8 decimals; on a piece
# with a 55 mm diagonal this is 5.5e-5 mm, under 1/3000 of its median edge
# (0.21 mm).
MERGE_TOLERANCE = 1e-6
# A separate component is debris when it has at most this many faces -- the
# cube has 12 -- or less than MIN_COMPONENT_SHARE of the area of the largest
# component with more. The cube is 0.4% to 4.9% of that component on the
# pieces checked, so the face count is the rule that catches it on small pieces;
# everything dropped on the Fractura pieces checked was the cube, and the area
# rule is for small loose parts elsewhere (Fantastic Breaks was not checked).
MAX_DEBRIS_FACES = 12
MIN_COMPONENT_SHARE = 0.01


@dataclass(frozen=True)
class CleanReport:
    """What :func:`clean_piece` changed, and the surface it left."""

    merged_vertices: int        # vertices that became another vertex
    collapsed_faces: int        # faces left with a repeated vertex by the merge
    duplicate_faces: int        # removed: interior walls cancelled, copies dropped
    dropped: Tuple[Tuple[int, float], ...]   # per dropped component: faces, area share
    flipped: bool               # wound inward, turned outward
    boundary_edges: int
    nonmanifold_edges: int
    components: int


class CleanPiece(NamedTuple):
    vertices: np.ndarray
    faces: np.ndarray
    face_labels: Optional[np.ndarray]     # the input's, carried to the kept faces
    report: CleanReport


def merge_close_vertices(vertices: np.ndarray, faces: np.ndarray, tolerance: float):
    """
    Vertices closer than ``tolerance`` (absolute) become one, at the position
    of the lowest-indexed of them. Returns ``(vertices, faces, kept)``: the
    merged vertices in ascending order of that index, the faces re-indexed with
    those that collapsed removed, and the mask of the input faces kept.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(vertices)
    pairs = (cKDTree(vertices).query_pairs(tolerance, output_type="ndarray")
             if n and tolerance > 0 else np.zeros((0, 2), np.int64))
    if len(pairs) == 0:
        return vertices, faces, np.ones(len(faces), dtype=bool)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, group = connected_components(graph, directed=False)
    # The lowest index of each group represents it; new indices follow that order.
    lowest = np.full(group.max() + 1, n, dtype=np.int64)
    np.minimum.at(lowest, group, np.arange(n))
    order = np.argsort(lowest)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    remapped = rank[group][faces]
    kept = ((remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2])
            & (remapped[:, 0] != remapped[:, 2]))
    return vertices[lowest[order]], remapped[kept], kept


def face_components(faces: np.ndarray, num_vertices: int):
    """``(count, label per face)``: faces are connected when they share a vertex."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    count = len(faces)
    if count == 0:
        return 0, np.zeros(0, dtype=np.int64)
    incidence = csr_matrix((np.ones(3 * count), (np.repeat(np.arange(count), 3),
                                                 np.asarray(faces).ravel())),
                           shape=(count, num_vertices))
    components, labels = connected_components(incidence @ incidence.T, directed=False)
    return components, labels


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    corners = np.asarray(vertices)[np.asarray(faces)]
    return 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                         corners[:, 2] - corners[:, 0]), axis=1)


def edge_counts(faces: np.ndarray) -> Tuple[int, int]:
    """``(boundary, non-manifold)``: edges with one face, and with more than two."""
    faces = np.asarray(faces)
    if len(faces) == 0:
        return 0, 0
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                    axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts > 2).sum())


def clean_piece(vertices, faces, face_labels=None, *,
                merge_tolerance: float = MERGE_TOLERANCE,
                max_debris_faces: int = MAX_DEBRIS_FACES,
                min_component_share: float = MIN_COMPONENT_SHARE,
                drop_debris: bool = True) -> CleanPiece:
    """
    One piece as one closed surface, in this order -- the duplicates only
    become duplicates once their vertices are shared:

    1. merge vertices closer than ``merge_tolerance`` x the bounding-box
       diagonal, and drop the faces that collapse;
    2. cancel opposite duplicate faces and drop repeated ones
       (:func:`~reassembly.mesh.repair.resolve_duplicated_faces`, the Breaking
       Bad reader's rule);
    3. keep the largest component by area of those with more than
       ``max_debris_faces`` faces, and every other one that is not debris
       (more than ``max_debris_faces`` faces and at least
       ``min_component_share`` of that one's area) -- skipped with
       ``drop_debris=False``, which keeps every component, as the Breaking
       Bad reader does;
    4. a closed surface wound inward is turned outward.

    ``face_labels`` (per input face, e.g. Fractura's break faces) come back for
    the faces kept.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    labels = None if face_labels is None else np.asarray(face_labels)
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0))) if len(vertices) else 0.0

    count = len(vertices)
    vertices, faces, kept = merge_close_vertices(vertices, faces, merge_tolerance * diagonal)
    merged, collapsed = count - len(vertices), int((~kept).sum())
    if labels is not None:
        labels = labels[kept]

    before = len(faces)
    faces, index = resolve_duplicated_faces(faces)
    faces = np.asarray(faces, dtype=np.int64)
    if labels is not None:
        labels = labels[index]
    duplicates = before - len(faces)
    used, faces = compact_indices(faces, len(vertices))
    vertices = vertices[used]

    dropped = []
    components, component = face_components(faces, len(vertices))
    if drop_debris and components > 1:
        area = np.bincount(component, weights=face_areas(vertices, faces),
                           minlength=components)
        size = np.bincount(component, minlength=components)
        # The reference is the largest of the components with more faces than
        # debris has: on a piece of a few square millimetres the cube is the
        # largest by area.
        sized = np.flatnonzero(size > max_debris_faces)
        candidates = sized if len(sized) else np.arange(components)
        largest = int(candidates[np.argmax(area[candidates])])
        debris = [c for c in range(components) if c != largest and (
            size[c] <= max_debris_faces or area[c] < min_component_share * area[largest])]
        if debris:
            dropped = [(int(size[c]), float(area[c] / area[largest])) for c in debris]
            keep = ~np.isin(component, debris)
            faces = faces[keep]
            if labels is not None:
                labels = labels[keep]
            used, faces = compact_indices(faces, len(vertices))
            vertices = vertices[used]
            components -= len(debris)

    boundary, nonmanifold = edge_counts(faces)
    flipped = False
    if len(faces) and boundary == 0:
        corners = vertices[faces] - vertices.mean(axis=0)
        volume = np.einsum("ij,ij->i", corners[:, 0],
                           np.cross(corners[:, 1], corners[:, 2])).sum() / 6.0
        if volume < 0:
            faces = faces[:, ::-1].copy()
            flipped = True

    report = CleanReport(merged, collapsed, duplicates, tuple(dropped), flipped,
                         boundary, nonmanifold, int(components))
    return CleanPiece(vertices, faces, labels, report)


def same_mesh(a_vertices, a_faces, b_vertices, b_faces) -> bool:
    """Whether two pieces are one mesh stored twice (the same vertex set and face count)."""
    a, b = np.asarray(a_vertices), np.asarray(b_vertices)
    if a.shape != b.shape or len(a_faces) != len(b_faces):
        return False
    return bool(np.array_equal(a[np.lexsort(a.T)], b[np.lexsort(b.T)]))


def identical_pieces(pieces: Sequence) -> list:
    """Every pair ``(i, j)`` of ``(vertices, faces)`` pieces that are the same mesh."""
    return [(i, j) for i in range(len(pieces)) for j in range(i + 1, len(pieces))
            if same_mesh(pieces[i][0], pieces[i][1], pieces[j][0], pieces[j][1])]


def copies_of_pieces(pieces: Sequence, max_debris_faces: int = MAX_DEBRIS_FACES,
                     components: Optional[Sequence[int]] = None) -> list:
    """
    ``(i, mask, j)`` for every component of a piece ``i`` that is piece ``j``'s
    surface stored again -- ``mask`` the component's faces in ``i``. A piece's
    surface is its largest component by area of those with more than
    ``max_debris_faces`` faces, as in :func:`clean_piece`, and only pieces with
    at least two such components are searched: debris is never taken for a
    copy, and the piece that was copied keeps its surface. Fractura's ``vert8``
    is one: its second piece (78,622 faces) is also one of its first piece's
    two components, in place and facing the same way.

    ``components``, each piece's count when it is known (:class:`CleanReport`),
    spares labelling the pieces that are one component.
    """
    parts = []
    for k, (vertices, faces) in enumerate(pieces):
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        if components is not None and components[k] == 1 and len(faces):
            count, component = 1, np.zeros(len(faces), dtype=np.int64)
        else:
            count, component = face_components(faces, len(vertices))
        size = np.bincount(component, minlength=count)
        sized = np.flatnonzero(size > max_debris_faces)
        candidates = sized if len(sized) else np.arange(count)
        if len(candidates) > 1:
            area = np.bincount(component, weights=face_areas(vertices, faces), minlength=count)
            main = int(candidates[np.argmax(area[candidates])])
        else:
            main = int(candidates[0]) if len(candidates) else -1
        parts.append((vertices, faces, component, sized, main, int(size[main]) if count else 0))

    def surface(k, c):
        vertices, faces, component = parts[k][:3]
        mask = component == c
        return vertices[np.unique(faces[mask])], faces[mask], mask

    found = []
    for i, (_, _, _, sized, _, _) in enumerate(parts):
        if len(sized) < 2:
            continue
        for c in sized:
            vertices, faces, mask = surface(i, c)
            for j, (_, _, _, _, main, main_faces) in enumerate(parts):
                if j == i or main < 0 or main_faces != len(faces):
                    continue
                if same_mesh(vertices, faces, *surface(j, main)[:2]):
                    found.append((i, mask, j))
                    break
    return found

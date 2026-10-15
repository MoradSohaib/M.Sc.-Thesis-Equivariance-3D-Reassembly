"""
Thinning fragment meshes to the density the model was trained at.

The network's five intra-fragment layers pass messages one mesh edge each, and
the dihedral fracture test compares neighbouring faces, so both see a surface
at the scale of its edges. Measured as the median edge over the object's
largest extent: seven Breaking Bad scenes 0.0064 to 0.0230 (median 0.0145);
Fractura, cleaned, its synthetic bones (simulated as Breaking Bad's are) 0.0092
to 0.0118 on three objects and its real bones 0.0015 to 0.0052 on seven --
edges 1.2 to 9.9 times shorter than that median, so the same five hops cover
that much less of the surface.

:func:`thin_assembly` gives one object a single face budget and splits it over
the pieces by surface area, so every piece ends at the same density and the
pieces keep their relative sizes -- not one count per piece, which would give a
small piece a large one's resolution. The thinning is quadric edge collapse
(``fast_simplification``, the Garland-Heckbert quadric error metric behind
trimesh's ``simplify_quadric_decimation``): each collapse puts the merged
vertex where it moves the surface least, so faces grow and the surface stays
where it was. Flat regions lose vertices first and creases keep them.

What it cannot keep is the dihedral labelling of the full-resolution scan. The
test is whether neighbouring faces differ by more than about 26 degrees, and
the same rough surface cut into larger triangles shows larger angles: the
fraction of faces it flags is a property of the resolution, not only of the
surface. The thing to match is the density the model was trained at, and the
converter reports the flagged fraction before and after.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Thinning starts only below this fraction of the target edge -- Breaking Bad's
# own scenes span a factor of 3.6 in median edge, so a mesh within a factor of
# two of the target is left alone (and keeps any vertices it shares with its
# neighbours, which thinning would take away).
TOLERANCE = 2.0
# No piece is thinned below this many faces.
MIN_FACES = 20
# Corrected (up to CORRECTIONS more passes) while the result's median edge
# misses the target by more than this.
EDGE_SLACK = 0.05
CORRECTIONS = 4
# Equilateral triangles of edge e cover sqrt(3)/4 e^2 each.
_TRIANGLE = np.sqrt(3.0) / 4.0


@dataclass(frozen=True)
class ThinReport:
    """What :func:`thin_assembly` did to one object."""

    thinned: bool
    edge_before: float        # median edge / the object's largest extent
    edge_after: float
    faces_before: int
    faces_after: int
    budget: int               # the object's face budget; 0 when not thinned


def object_extent(pieces: Sequence) -> float:
    """The largest side of the bounding box of all the pieces together."""
    points = np.concatenate([np.asarray(v, dtype=np.float64) for v, _ in pieces])
    return float(np.ptp(points, axis=0).max())


def edge_lengths(vertices, faces) -> np.ndarray:
    corners = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    return np.concatenate([np.linalg.norm(corners[:, i] - corners[:, (i + 1) % 3], axis=1)
                           for i in range(3)])


def median_edge(pieces: Sequence) -> float:
    """Median edge length over every face of every piece (an edge per face side)."""
    return float(np.median(np.concatenate([edge_lengths(v, f) for v, f in pieces])))


def surface_area(vertices, faces) -> float:
    corners = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    return float(0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                               corners[:, 2] - corners[:, 0]), axis=1).sum())


def area_budget(areas: Sequence[float], total: float, counts: Sequence[int],
                minimum: int = MIN_FACES) -> List[int]:
    """``total`` faces split by area, each at least ``minimum`` and at most the piece's own."""
    areas = np.asarray(areas, dtype=np.float64)
    share = areas / areas.sum() if areas.sum() > 0 else np.full(len(areas), 1.0 / len(areas))
    return [int(min(count, max(minimum, round(total * s)))) for s, count in zip(share, counts)]


def simplify(vertices, faces, target: int, aggression: float = 7.0):
    """Quadric edge collapse to about ``target`` faces; the mesh unchanged if it has fewer."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if target >= len(faces):
        return vertices, faces
    try:
        import fast_simplification
    except ImportError as error:                    # pragma: no cover - environment
        raise ImportError("thinning needs fast_simplification: pip install "
                          "fast-simplification") from error
    points, triangles = fast_simplification.simplify(vertices, faces, target_count=int(target),
                                                     agg=aggression)
    return np.asarray(points, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def thin_assembly(pieces: Sequence, *, edge: Optional[float] = None,
                  faces: Optional[int] = None, tolerance: float = TOLERANCE,
                  minimum: int = MIN_FACES) -> Tuple[list, ThinReport]:
    """
    ``pieces`` (``(vertices, faces)`` each, one object) thinned to a density.

    ``edge``: the target median edge as a fraction of the object's largest
    extent; the budget is the surface area over the area of an equilateral
    triangle of that edge, then searched (at most four more passes, secant
    steps in log-log) while the result misses the target by more than 5%; the
    closest pass is kept. ``faces``: a fixed budget for the object. Either way the
    budget is split by area (:func:`area_budget`), and an object is thinned
    only when it is denser than the target by more than ``tolerance`` (in
    edge length; for ``faces``, when it has more than ``tolerance**2`` times
    the budget). Neither: returned as it is.
    """
    pieces = [(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64))
              for v, f in pieces]
    extent = object_extent(pieces)
    before = median_edge(pieces) / extent if extent > 0 else 0.0
    count = sum(len(f) for _, f in pieces)
    unchanged = ThinReport(False, before, before, count, count, 0)
    if extent <= 0 or (edge is None and faces is None):
        return pieces, unchanged

    areas = [surface_area(v, f) for v, f in pieces]
    counts = [len(f) for _, f in pieces]
    if faces is not None:
        if count <= faces * tolerance ** 2:
            return pieces, unchanged
        budget = float(faces)
    else:
        if before >= edge / tolerance:
            return pieces, unchanged
        budget = sum(areas) / (_TRIANGLE * (edge * extent) ** 2)

    def run(total: float):
        targets = area_budget(areas, total, counts, minimum)
        thinned = [simplify(v, f, t) for (v, f), t in zip(pieces, targets)]
        return thinned, median_edge(thinned) / extent

    tried = [(budget, *run(budget))]
    for _ in range(CORRECTIONS if edge is not None else 0):
        if abs(tried[-1][2] / edge - 1.0) <= EDGE_SLACK:
            break
        if len(tried) == 1:            # the area formula's own exponent: edge ~ budget^-1/2
            slope = -0.5
        else:
            (b0, _, e0), (b1, _, e1) = tried[-2], tried[-1]
            slope = (np.log(e1) - np.log(e0)) / (np.log(b1) - np.log(b0)) \
                if b1 != b0 and e1 != e0 else -0.5
            slope = min(slope, -0.05)  # finer when the budget grows, never flat
        last, _, achieved = tried[-1]
        budget = float(np.exp(np.log(last) + (np.log(edge) - np.log(achieved)) / slope))
        tried.append((budget, *run(budget)))
    budget, thinned, after = min(tried, key=lambda t: abs(t[2] / edge - 1.0)) \
        if edge is not None else tried[-1]
    return thinned, ThinReport(True, before, after, count, sum(len(f) for _, f in thinned),
                               int(round(budget)))


def transfer_face_labels(source_vertices, source_faces, labels, vertices, faces) -> np.ndarray:
    """Each face's label from the source face whose centroid is nearest its own."""
    from scipy.spatial import cKDTree

    source = np.asarray(source_vertices, dtype=np.float64)[np.asarray(source_faces)].mean(axis=1)
    target = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)].mean(axis=1)
    return np.asarray(labels)[cKDTree(source).query(target)[1]]

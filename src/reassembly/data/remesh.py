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
    ``[low, high]``, then every vertex moves by Gaussian noise of ``jitter``."""

    low: float
    high: float
    jitter: float = 0.0


def check_remesh(low: float, high: float, jitter: float = 0.0) -> None:
    """Refuse a setting that would not remesh, or could not."""
    if not 0.0 < low <= high <= 1.0:
        raise ValueError(f"remesh keeps a share of each piece's faces: LOW and HIGH must "
                         f"satisfy 0 < LOW <= HIGH <= 1, got {low:g} and {high:g}")
    if jitter < 0.0:
        raise ValueError(f"remesh_jitter is a standard deviation: >= 0, got {jitter:g}")


def remesh_pieces(vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray],
                  rng: np.random.Generator, spec: Remesh
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    ``(vertices, faces)`` per piece, each remeshed on its own (module
    docstring). The shares are drawn first, one per piece, so a piece's share
    does not depend on the pieces before it; then each piece is collapsed and,
    with ``jitter``, moved. A piece of :data:`~reassembly.mesh.decimate.MIN_FACES`
    faces or fewer keeps its faces, and so does one the collapse would leave
    empty; the jitter moves every piece. The collapse is tidied as the
    converter tidies it -- vertices merged, the faces that collapse and
    repeated faces removed (:func:`~reassembly.mesh.clean.clean_piece`), every
    component kept -- and keeps the winding the collapse leaves, the piece's
    own.
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
        if spec.jitter:
            v = v + rng.normal(0.0, spec.jitter, v.shape)
        out_vertices.append(v)
        out_faces.append(f)
    return out_vertices, out_faces


__all__ = ["Remesh", "check_remesh", "remesh_pieces"]

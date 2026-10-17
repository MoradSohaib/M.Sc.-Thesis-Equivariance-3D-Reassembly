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

It does not open the breaks. Remeshed, the two sides still lie within a
fraction of an edge of each other -- a median gap of 0.003 to 0.012 among the
vertices within 0.05 of another piece -- where Fractura's real pieces, as
converted, sit 0.013 to 0.029 apart.

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


def remesh_pieces(vertices: Sequence[np.ndarray], faces: Sequence[np.ndarray],
                  rng: np.random.Generator, spec: Remesh
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    ``(vertices, faces)`` per piece, each remeshed on its own (module
    docstring): :func:`collapse_pieces`, then :func:`jitter_pieces` with
    ``spec.jitter``, from the same generator. The jitter moves the vertices,
    never the faces.
    """
    vertices, faces = collapse_pieces(vertices, faces, rng, spec)
    return jitter_pieces(vertices, rng, spec.jitter), faces


__all__ = ["Remesh", "check_remesh", "collapse_pieces", "jitter_pieces", "remesh_pieces"]

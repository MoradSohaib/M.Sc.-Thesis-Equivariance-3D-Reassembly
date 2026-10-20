"""
Two layers that read a whole fragment at once: the fracture head, which says
which vertices lie on a break and so where the cross-fragment tokens go, and
the fragment context, which tells every vertex what its whole piece looks like.

The fracture head
-----------------
v7 chose the tokens -- and the vertices stage two matches -- with a dihedral
test on the mesh: a sharp crease marks a break. On Breaking Bad it over-labels
3.9x (precision 0.24, recall 0.93), and on a scan it floods: a jitter of 0.001
makes it flag 90% to 99.5% of the vertices, and the scanned pieces of Fractura
are flagged at 29% to 51% of their faces once thinned. The head learns the
break instead, from the vertices a Breaking Bad piece shares with another
(carried over to a remeshed piece), and the network picks its tokens from what
it predicts, on the device (:func:`select_tokens`).

It reads the features as invariants (:class:`~reassembly.nn.vn.VNInvariant`),
so its answer does not change when the fragment is turned: whether a vertex is
on a break is a property of the shape, not of its pose.

The fragment context
--------------------
A vertex's features describe as many rings of the mesh as there are intra
layers before it -- a few edges. :class:`VNFragmentContext` pools the whole
fragment's features into one equivariant summary and hands it back to every
vertex. The pool is a mean of vectors that all turn with the fragment, so it
turns with the fragment too, and nothing of any other fragment enters it: the
layer is equivariant to its own fragment's pose and blind to the others'.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .segment import segment_mean
from .vn import VNInvariant, VNLayerNorm, VNLeakyReLU, VNLinear


class FractureHead(nn.Module):
    """Per-vertex logit of being on a break, from the vector features' invariants."""

    def __init__(self, channels: int, hidden: int = 64, directions: int = 4) -> None:
        super().__init__()
        self.invariant = VNInvariant(channels, directions=directions)
        self.mlp = nn.Sequential(
            nn.Linear(self.invariant.out_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """``(N, C, 3) -> (N,)`` logits."""
        return self.mlp(self.invariant(x)).squeeze(-1)


class VNFragmentContext(nn.Module):
    """
    ``x + W_o act(W_x x + W_g mean_fragment(x))``, normalised: every vertex hears
    its whole fragment. Starts near the identity -- ``W_o`` small -- so adding
    the layer to a trained schedule does not scramble it.
    """

    def __init__(self, channels: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.own = VNLinear(channels, channels)
        self.pooled = VNLinear(channels, channels)
        self.act = VNLeakyReLU(channels, negative_slope)
        self.out = VNLinear(channels, channels)
        self.norm = VNLayerNorm(channels)
        # Small, not zero: a zero output weight would leave every weight before
        # it without a gradient until it moved (the cross layer's reasoning).
        nn.init.uniform_(self.out.weight, -0.01 / math.sqrt(channels), 0.01 / math.sqrt(channels))

    def forward(self, x: Tensor, vertex_fragment: Tensor, num_fragments: int) -> Tensor:
        pooled = segment_mean(x, vertex_fragment, num_fragments)          # (F, C, 3)
        hidden = self.act(self.own(x) + self.pooled(pooled)[vertex_fragment])
        return self.norm(x + self.out(hidden))


# A fragment whose head predicts no break at all still gets this many tokens,
# its most likely vertices, so it is not cut off from the others.
FALLBACK_TOKENS = 8
# Two distances within this share of each other are a tie, broken by vertex
# order -- which a rotation does not change, where round-off would: on a
# symmetric piece many vertices lie at one distance, and float noise of 1e-16
# would otherwise pick a different one in every pose.
TIE_TOLERANCE = 1e-9


def _allocate(counts: Tensor, budget: int) -> Tensor:
    """
    ``budget`` split over fragments in proportion to ``counts``, each at least
    1 if it has any and at most its count -- largest remainders first. ``(F,)``.
    """
    counts = counts.long()
    total = int(counts.sum())
    if total == 0 or budget <= 0:
        return torch.zeros_like(counts)
    if total <= budget:
        return counts.clone()
    share = counts.double() * budget / total
    out = torch.floor(share).long()
    out = torch.where((counts > 0) & (out == 0), torch.ones_like(out), out)
    out = torch.minimum(out, counts)
    left = budget - int(out.sum())
    if left > 0:
        remainder = torch.where(out < counts, share - torch.floor(share), torch.full_like(share, -1.0))
        order = torch.argsort(remainder, descending=True)
        for index in order.tolist():
            if left <= 0 or remainder[index] < 0:
                break
            out[index] += 1
            left -= 1
    elif left < 0:
        # The minimum of 1 can overshoot; take back from the largest shares.
        order = torch.argsort(out, descending=True)
        for index in order.tolist():
            if left >= 0:
                break
            if out[index] > 1:
                out[index] -= 1
                left += 1
    return out


def select_tokens(eligible: Tensor, points: Tensor, vertex_fragment: Tensor,
                  fragment_scene: Tensor, num_fragments: int, num_scenes: int,
                  budget: int, score: Optional[Tensor] = None) -> Tensor:
    """
    The cross-fragment tokens, picked on the device: ``budget`` per scene,
    split over its fragments by how many eligible vertices each has (on an
    evenly meshed piece, by the area of its break), then spread over each
    fragment's eligible vertices by farthest-point sampling on ``points``.
    Sorted, so each fragment's tokens are contiguous and the scenes in order,
    as :func:`~reassembly.nn.cross.cross_fragment_index` requires.

    ``score`` (``(N,)``, higher is likelier) makes it the predicted case: a
    fragment with no eligible vertex takes its :data:`FALLBACK_TOKENS` likeliest
    instead, so no fragment is cut off by a head that missed its break. With no
    ``score`` the mask is the truth and is taken as it is.

    The first point of each fragment is its eligible vertex farthest from
    their centroid, and every distance is within one fragment, so the choice
    does not change when a fragment is turned.
    """
    device = points.device
    eligible = eligible.bool().clone()
    if score is not None:
        counts = torch.bincount(vertex_fragment[eligible], minlength=num_fragments)
        for fragment in torch.nonzero(counts == 0).flatten().tolist():
            own = torch.nonzero(vertex_fragment == fragment).flatten()
            if own.numel():
                values = score[own].double()
                kth = torch.topk(values, min(FALLBACK_TOKENS, own.numel())).values[-1]
                # Every vertex tied with the last one taken, so the set does
                # not depend on round-off (TIE_TOLERANCE).
                eligible[own[values >= kth - TIE_TOLERANCE * kth.abs().clamp(min=1.0)]] = True
    index = torch.nonzero(eligible).flatten()
    if index.numel() == 0:
        return index
    fragment = vertex_fragment[index]
    counts = torch.bincount(fragment, minlength=num_fragments)
    quota = torch.zeros(num_fragments, dtype=torch.long, device=device)
    for scene in range(num_scenes):
        own = torch.nonzero(fragment_scene == scene).flatten()
        if own.numel():
            quota[own] = _allocate(counts[own], budget)

    p = points[index].double()
    # A fragment whose quota covers every eligible vertex takes them all; the
    # rest are sampled, from the eligible vertex farthest from their centroid.
    whole = quota >= counts
    chosen_mask = whole[fragment].clone()
    quota = torch.where(whole, torch.zeros_like(quota), quota)
    centre = segment_mean(p, fragment, num_fragments)
    nearest = (p - centre[fragment]).norm(dim=-1)
    position = torch.arange(index.numel(), device=device)
    steps = int(quota.max()) if quota.numel() else 0
    first = True
    for step in range(steps):
        active = quota > step                                            # (F,)
        live = active[fragment] & ~chosen_mask
        if not bool(live.any()):
            break
        score_now = torch.where(live, nearest, torch.full_like(nearest, -1.0))
        best = torch.full((num_fragments,), -1.0, dtype=score_now.dtype, device=device)
        best = best.scatter_reduce(0, fragment, score_now, reduce="amax", include_self=True)
        top = best[fragment]
        hit = live & (score_now >= top - TIE_TOLERANCE * top.abs().clamp(min=1e-12))
        # One per fragment: the first of any tie.
        pick = torch.full((num_fragments,), index.numel(), dtype=torch.long, device=device)
        pick = pick.scatter_reduce(0, fragment[hit], position[hit], reduce="amin",
                                   include_self=True)
        picked = pick[active & (pick < index.numel())]
        chosen_mask[picked] = True
        owner = torch.full((num_fragments,), -1, dtype=torch.long, device=device)
        owner[fragment[picked]] = picked
        anchor = owner[fragment]
        has = anchor >= 0
        distance = torch.where(has, (p - p[anchor.clamp(min=0)]).norm(dim=-1), nearest)
        nearest = distance if first else torch.minimum(nearest, distance)
        first = False
    return torch.sort(index[chosen_mask]).values


def segment_loss(logits: Tensor, target: Tensor, vertex_fragment: Tensor,
                 num_fragments: int, gamma: float = 2.0,
                 keep: Optional[Tensor] = None) -> Tensor:
    """
    The fracture head's loss: focal binary cross-entropy (``gamma``; 0 is
    plain cross-entropy), each fragment's breaks and non-breaks weighted to
    count equally -- a break is about a tenth of the vertices -- then the mean
    over each fragment's vertices and over the fragments ``keep`` (``(N,)``
    bool, the vertices whose truth is known) leaves any.

    Reads 0.693 at ``p = 0.5`` with ``gamma = 0`` and a quarter of that,
    0.173, with ``gamma = 2``: the weights average to 1 over a fragment.
    """
    target = target.to(logits.dtype)
    if keep is None:
        keep = torch.ones_like(target, dtype=torch.bool)
    if not bool(keep.any()):
        return logits.new_zeros(())
    logits, target, fragment = logits[keep], target[keep], vertex_fragment[keep]
    count = torch.bincount(fragment, minlength=num_fragments).to(logits.dtype)
    positive = torch.bincount(fragment, weights=target, minlength=num_fragments)
    share = (positive / count.clamp(min=1.0)).clamp(0.01, 0.99)
    weight = torch.where(target > 0.5, 0.5 / share[fragment], 0.5 / (1.0 - share[fragment]))
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if gamma:
        p = torch.sigmoid(logits)
        p_true = torch.where(target > 0.5, p, 1.0 - p)
        bce = bce * (1.0 - p_true) ** gamma
    per_fragment = segment_mean((weight * bce).unsqueeze(-1), fragment, num_fragments).squeeze(-1)
    present = count > 0
    return per_fragment[present].mean()


def segment_scores(predicted: Tensor, target: Tensor,
                   keep: Optional[Tensor] = None) -> Tuple[float, float]:
    """``(precision, recall)`` of a predicted break mask against the truth."""
    if keep is not None:
        predicted, target = predicted[keep], target[keep]
    predicted, target = predicted.bool(), target.bool()
    hit = float((predicted & target).sum())
    chosen, marked = float(predicted.sum()), float(target.sum())
    return (hit / chosen if chosen else float("nan"), hit / marked if marked else float("nan"))

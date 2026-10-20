"""
The backbone: intra-fragment attention, cross-fragment attention, and the
per-vertex embedding the fragments are matched by.

Shape of the network
--------------------
Layers alternate. Intra-fragment attention moves information along real mesh
edges; cross-fragment attention lets a fragment's interface tokens hear what
the other fragments in its scene look like; then more intra-fragment layers
carry that back through the mesh. A cross layer only updates the *token*
vertices, so an intra layer after it is what spreads what they heard to their
neighbours.

The default schedule is ``intra, intra, cross, intra, cross, intra, cross,
intra`` -- three rounds of "describe the fragment, then let the fragments
talk", each followed by one hop along the mesh. ``intra`` alone, five times, is
the ablation without any cross-fragment layer.

Two more kinds (``nn/context.py``): ``segment``, the fracture head -- once, and
before every ``cross`` -- which predicts per vertex whether it lies on a break;
the network then picks its own tokens from that prediction, on the device, and
stage two matches the vertices it predicts (``training._forward``). And
``global``, which hands every vertex a summary of its whole fragment. For
example ``intra intra intra intra segment global cross intra cross intra cross
intra``: four hops of mesh context before the head reads it.

What comes out
--------------
An invariant embedding per vertex (``readout`` + ``embedding``), trained by the
contrastive term in ``nn/losses.py`` to make the two sides of a break look the
same, and the equivariant features it is read from. Nothing else: the rotation
head this network used to end in -- a pooled frame per fragment, made
orthonormal by Gram-Schmidt -- was removed in v7. Trained, its rotations stayed
far from usable on shapes it had not seen: 102 deg anchor-aligned on Everyday's
validation scenes (W10, epoch 334), where the rotations fitted from these
embeddings' matches were 15 deg off. A run trained on the embedding term alone
(nrhl) did better than W10 on Everyday's validation split -- 10.2 deg against
15.2, acc@5 0.905 against 0.847 -- so training the head bought nothing.
Each fragment's rotation now comes from stage two (``assembly/rotation.py``);
``docs/PROJECT-STATE.md`` has the measurements.

The rotation convention
-----------------------
A rotation here maps the perturbed fragment back to its assembled pose,
applied to row vectors: ``v_pert @ R.T == v_assembled`` (``nn/losses.py`` and
:func:`apply_rotation`). Centred, the perturbation is ``v_pert = v_gt Q^T``, so
the label is ``R_gt = Q^T``. The fitted rotations of stage two use the same
convention.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from .context import FractureHead, VNFragmentContext, select_tokens
from .cross import VNCrossFragmentAttention, cross_fragment_index
from .gat import VNGraphAttentionBlock
from .schedule import LAYER_KINDS, check_schedule  # noqa: F401  (LAYER_KINDS re-exported)
from .vn import VNInvariant, VNLinear, VNScaleGate

DEFAULT_SCHEDULE = ("intra", "intra", "cross", "intra", "cross", "intra", "cross", "intra")


class Prediction(NamedTuple):
    """What one forward pass produces."""
    vertex_embedding: Tensor    # (N, D) invariant: what stage two matches, and
    #                             what the contrastive loss trains
    vertex_features: Tensor     # (N, C, 3) equivariant, the backbone's output
    fracture_logit: Optional[Tensor] = None   # (N,) the fracture head's, with a
    #                                           segment layer; else None
    token_index: Optional[Tensor] = None      # (T,) the tokens the cross layers
    #                                           used, when the network picked them


class ReassemblyNet(nn.Module):
    """
    SO(3)-equivariant per-vertex features for fractured fragments, read out as
    an invariant embedding for matching the fragments' break surfaces.

    Every fragment is already a graph -- its mesh -- so nothing is constructed
    inside one. The only built connections are between fragments, among the
    sampled fracture-surface tokens.
    """

    def __init__(
        self,
        channels: int = 32,
        node_channels: int = 2,
        edge_channels: int = 3,
        heads: int = 4,
        head_dim: int = 8,
        embedding_dim: int = 32,
        schedule: Sequence[str] = DEFAULT_SCHEDULE,
        negative_slope: float = 0.2,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.schedule = tuple(schedule)
        check_schedule(self.schedule)

        self.embed = VNLinear(node_channels, channels)
        # Fragment scale joins as an invariant gate, never as a third axis of a
        # (C, 3) tensor -- scale has no direction, so there is no legal slot.
        self.scale_gate = VNScaleGate(channels, scalar_features=1)

        self.intra = nn.ModuleList(
            VNGraphAttentionBlock(channels, edge_channels, heads, head_dim,
                                  negative_slope)
            for kind in self.schedule if kind == "intra"
        )
        # The cross layers always recompute their pair gathers in the backward
        # pass (1109 -> 86 bytes per pair, bitwise-identical results, nearly
        # free): not a setting, because there is no reason to turn it off.
        self.cross = nn.ModuleList(
            VNCrossFragmentAttention(channels, heads=heads, head_dim=2 * head_dim)
            for kind in self.schedule if kind == "cross"
        )
        # Absent, not empty, without the kinds: a checkpoint from before they
        # existed holds no key for them and loads exactly as it did.
        self.fracture_head = FractureHead(channels) if "segment" in self.schedule else None
        if "global" in self.schedule:
            self.context = nn.ModuleList(
                VNFragmentContext(channels, negative_slope)
                for kind in self.schedule if kind == "global")
        self.grad_checkpointing = bool(grad_checkpointing)
        """
        Recompute every layer in the backward pass instead of storing its
        insides, as Thesis 1's ``--grad_checkpointing`` does. Each layer then
        keeps only its input, ``(N, C, 3)``; without it an intra layer keeps
        several edge-sized ``(E, C, 3)`` tensors, and a mesh has about six
        directed edges per vertex. Outputs and gradients are identical either
        way; the price is roughly one extra forward pass. Read at call time, so
        the out-of-memory retry can switch it on for one batch.
        """

        # Invariant per-vertex embedding for matching interface points across
        # fragments. Invariant, not equivariant: the vertices being matched sit
        # in differently-perturbed fragments, so equivariant features would be
        # compared across unrelated frames.
        self.readout = VNInvariant(channels, directions=4)
        self.embedding = nn.Sequential(
            nn.Linear(self.readout.out_features, 2 * embedding_dim),
            nn.LayerNorm(2 * embedding_dim),
            nn.GELU(),
            # No bias on the last layer, and this is provable rather than
            # stylistic. The embedding is supervised only by the
            # contrastive loss, and a constant added to every embedding
            # shifts them all equally -- the scatter about each cluster centroid
            # does not move. It is unidentifiable for stage two as well, since a
            # global shift leaves every pairwise distance untouched. Keeping it
            # would put a parameter in the model that provably cannot be
            # learned, which then shows up forever as a "dead gradient".
            nn.Linear(2 * embedding_dim, embedding_dim, bias=False),
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        vertex_fragment: Tensor,
        num_fragments: int,
        *,
        log_scale: Optional[Tensor] = None,
        token_index: Optional[Tensor] = None,
        token_query: Optional[Tensor] = None,
        token_key: Optional[Tensor] = None,
        fragment_scene: Optional[Tensor] = None,
        num_scenes: Optional[int] = None,
        token_mask: Optional[Tensor] = None,
        tokens_per_scene: int = 2048,
        threshold: float = 0.5,
    ) -> Prediction:
        """
        ``node_features`` ``(N, node_channels, 3)`` -- centred coordinate and
        vertex normal. ``edge_attr`` ``(E, edge_channels, 3)`` -- the two
        canonically ordered face normals and the relative position of the
        source vertex. ``token_index`` selects which vertices
        take part in cross-fragment attention; ``token_query`` and ``token_key``
        are the pair lists from :func:`~reassembly.nn.cross.cross_fragment_index`,
        built once per batch. ``num_fragments`` counts the fragments, for the
        ``global`` layers.

        With a ``segment`` layer the network picks its tokens itself, there,
        and the three token arguments are ignored: ``tokens_per_scene`` of them
        per scene (:func:`~reassembly.nn.context.select_tokens`) among the
        vertices whose predicted probability of a break is above ``threshold``
        -- or, given ``token_mask`` (``(N,)`` bool), among those: the true
        break, to train the cross layers on while the head is learning, or the
        oracle. It needs ``fragment_scene`` and ``num_scenes``.
        """
        x = self.embed(node_features)
        if log_scale is not None:
            x = self.scale_gate(x, log_scale[vertex_fragment])

        checkpointed = self.grad_checkpointing and torch.is_grad_enabled()

        def run(layer, *inputs):
            if checkpointed:
                return torch.utils.checkpoint.checkpoint(layer, *inputs,
                                                         use_reentrant=False)
            return layer(*inputs)

        intra = iter(self.intra)
        cross = iter(self.cross)
        context = iter(self.context) if "global" in self.schedule else iter(())
        fracture_logit = picked = None
        for kind in self.schedule:
            if kind == "intra":
                x = run(next(intra), x, edge_index, edge_attr)
            elif kind == "global":
                x = run(next(context), x, vertex_fragment, num_fragments)
            elif kind == "segment":
                if fragment_scene is None or num_scenes is None:
                    raise ValueError("a segment layer picks the tokens per scene: pass "
                                     "fragment_scene and num_scenes")
                fracture_logit = self.fracture_head(x)
                with torch.no_grad():
                    if token_mask is not None:
                        eligible, score = token_mask.bool(), None
                    else:
                        eligible = torch.sigmoid(fracture_logit) > threshold
                        score = fracture_logit.detach()
                    token_index = select_tokens(
                        eligible, node_features[:, 0, :].detach(), vertex_fragment,
                        fragment_scene, num_fragments, int(num_scenes),
                        int(tokens_per_scene), score=score)
                    token_fragment = vertex_fragment[token_index]
                    token_query, token_key = cross_fragment_index(
                        token_fragment, fragment_scene[token_fragment], int(num_scenes))
                picked = token_index
            else:
                layer = next(cross)
                if token_index is None or token_index.numel() == 0:
                    # No tokens anywhere in the batch. The layer is a no-op
                    # rather than an error: a single-fragment mode has no
                    # cross-fragment structure to attend over.
                    continue
                tokens = run(layer, x[token_index], token_query, token_key)
                # Residual write-back, so vertices that are not tokens keep
                # their features and token vertices keep theirs too.
                x = x.index_add(0, token_index, tokens - x[token_index])

        return Prediction(
            vertex_embedding=self.embedding(self.readout(x)),
            vertex_features=x,
            fracture_logit=fracture_logit,
            token_index=picked,
        )

    @property
    def segments(self) -> bool:
        """Whether the network has a fracture head, and so picks its own tokens."""
        return self.fracture_head is not None


def apply_rotation(vertices: Tensor, rotation: Tensor,
                   vertex_fragment: Tensor) -> Tensor:
    """
    ``v @ R.T`` per fragment -- the operation the position and normal scores
    compare against the ground truth, and stage two places fragments with.
    """
    return torch.einsum("nij,nj->ni", rotation[vertex_fragment], vertices)

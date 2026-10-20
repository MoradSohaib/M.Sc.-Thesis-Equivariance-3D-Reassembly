"""
The layer kinds a schedule may name, and the rules for ordering them -- here,
without torch, so a ``Config`` can be checked on a machine that has none.
"""
from __future__ import annotations

from typing import Sequence

LAYER_KINDS = ("intra", "cross", "segment", "global")
"""``intra``: attention along the mesh (``nn/gat.py``); ``cross``: between the
fragments' tokens (``nn/cross.py``); ``segment``: the fracture head, which picks
the tokens; ``global``: each fragment's summary handed to its vertices
(``nn/context.py``)."""


def check_schedule(schedule: Sequence[str]) -> None:
    """Refuse a schedule the network cannot build: an unknown kind, two fracture
    heads, or a cross layer before the head that picks its tokens."""
    schedule = list(schedule)
    unknown = set(schedule) - set(LAYER_KINDS)
    if unknown:
        raise ValueError(f"unknown layer kinds in schedule: {sorted(unknown)} "
                         f"(the kinds are {', '.join(LAYER_KINDS)})")
    if schedule.count("segment") > 1:
        raise ValueError("schedule: one segment layer at most -- it picks the tokens once")
    if "segment" in schedule and "cross" in schedule[:schedule.index("segment")]:
        raise ValueError("schedule: every cross layer must come after segment, which picks "
                         "the tokens the cross layers attend between")

"""Actions and the progress-band mask (RL doc 2.3, 6.1).

Atomic actions (one decision step):
    TAKE       buy at the ask, up to SliceConfig.take_lots
    JOIN       rest a buy order at the best bid
    IMPROVE    rest at bid + 1 tick (only if the spread is 2+ ticks)
    POST_DEEP  rest at bid - 1 tick
    CANCEL     cancel the resting order
    HOLD       do nothing (keep the resting order, if any)

The progress band is a HARD mask, never a reward penalty:
    behind  (progress < L + eps)  -> only {TAKE, IMPROVE}
    ahead   (progress > U - eps)  -> everything except TAKE
    spread == 1                    -> no IMPROVE
    no resting order               -> no CANCEL
    slice already complete         -> only HOLD
Masks are boolean arrays of shape (n, N_ACTIONS); True = allowed. Every policy must apply them
(scores of masked actions set to -inf), and the logging policy's propensities are computed over the
allowed set only.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    TAKE = 0
    JOIN = 1
    IMPROVE = 2
    POST_DEEP = 3
    CANCEL = 4
    HOLD = 5


N_ACTIONS = len(Action)
ACTION_NAMES = [a.name for a in Action]


def action_mask(progress, lower, upper, spread_ticks, has_order, done, eps_band: float) -> np.ndarray:
    """Vectorised mask; every argument is an array of shape (n,) (scalars broadcast)."""
    progress, lower, upper = np.broadcast_arrays(np.asarray(progress, float), lower, upper)
    n = progress.shape[0]
    spread_ticks = np.broadcast_to(spread_ticks, (n,))
    has_order = np.broadcast_to(np.asarray(has_order, bool), (n,))
    done = np.broadcast_to(np.asarray(done, bool), (n,))
    m = np.ones((n, N_ACTIONS), bool)
    behind = progress < lower + eps_band
    ahead = progress > upper - eps_band
    m[behind] = False
    m[behind, Action.TAKE] = True
    m[behind, Action.IMPROVE] = True
    m[ahead, Action.TAKE] = False
    m[spread_ticks < 2, Action.IMPROVE] = False
    m[~has_order, Action.CANCEL] = False
    # behind with a 1-tick spread leaves TAKE only; make sure at least one action is always allowed
    m[done] = False
    m[done, Action.HOLD] = True
    empty = ~m.any(axis=1)
    m[empty, Action.TAKE] = True
    return m


def apply_mask(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scores of disallowed actions -> -inf, so argmax / softmax never pick them."""
    return np.where(mask, scores, -np.inf)


@dataclass(frozen=True)
class MacroAction:
    """A bandit arm (RL doc 6.1): place `kind` now, hold for `hold_steps`, then hand the rest to the
    clean-up rule (buy at the ask). TAKE with hold_steps=1 means "take now"."""
    name: str
    kind: Action
    hold_steps: int


DEFAULT_MACROS = (
    MacroAction("take_now", Action.TAKE, 1),
    MacroAction("join_hold2", Action.JOIN, 2),
    MacroAction("join_hold5", Action.JOIN, 5),
    MacroAction("improve_hold2", Action.IMPROVE, 2),
)


def macro_mask(spread_ticks, behind, macros=DEFAULT_MACROS) -> np.ndarray:
    """Allowed macros: IMPROVE needs a 2-tick spread; when behind schedule only TAKE / IMPROVE."""
    spread_ticks = np.asarray(spread_ticks)
    behind = np.asarray(behind, bool)
    m = np.ones((len(spread_ticks), len(macros)), bool)
    for j, mac in enumerate(macros):
        if mac.kind == Action.IMPROVE:
            m[:, j] &= spread_ticks >= 2
        if mac.kind not in (Action.TAKE, Action.IMPROVE):
            m[:, j] &= ~behind
    m[~m.any(axis=1), 0] = True          # TAKE is the fallback
    return m

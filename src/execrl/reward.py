"""Rewards derived from execution cost (RL doc 2.4 and 6.1). Pure functions, numpy in / numpy out.

Sign convention: cost C in bp is "bad when positive"; reward = -cost. Buy side d = +1, sell d = -1.

Slice cost relative to the slice-start mid m0 (fills plus the clean-up at the end):
    C = 1e4 / (q* m0) * [ d * sum_i v_i (p_i - m0)  +  d * q_rem_N (p_clean - m0) ]
Per-step reward r_n = cost of the fills inside step n (clean-up added to the last step), so
sum_n r_n = -C exactly.

Potential-based shaping (Ng et al. 1999), terminal potential 0:
    Phi(s_n) = -1e4 / (q* m0) * q_rem_n * (spread_n * tick / 2 + fee_take)   for n < N,  Phi(s_N) = 0
    r~_n = r_n + Phi(s_{n+1}) - Phi(s_n)        sum_n r~_n = -C - Phi(s_0)  (policy-independent)
"""
from __future__ import annotations

import numpy as np


def bp_scale(q_star, m0):
    """Multiplier turning (lots x price difference) into bp of the slice notional."""
    return 1e4 / (np.asarray(q_star, float) * np.asarray(m0, float))


def fill_cost(v, p, m_ref, d: int = 1):
    """d * v * (p - m_ref): the currency cost of one fill (or an array of fills) vs a reference mid."""
    return d * np.asarray(v, float) * (np.asarray(p, float) - np.asarray(m_ref, float))


def slice_cost_bp(fill_v, fill_p, m0, q_star, q_rem_end, p_clean, d: int = 1) -> float:
    """C for ONE slice from its fills (arrays) and the clean-up (scalars)."""
    c = np.sum(fill_cost(fill_v, fill_p, m0, d)) + fill_cost(q_rem_end, p_clean, m0, d)
    return float(bp_scale(q_star, m0) * c)


def step_reward(step_fill_cost, q_star, m0):
    """r_n = -bp_scale * (sum of fill costs in step n). Vectorised over envs."""
    return -bp_scale(q_star, m0) * np.asarray(step_fill_cost, float)


def potential(q_rem, q_star, m0, spread_ticks, tick, fee_take, terminal):
    """Phi(s): minus the bp cost of taking everything left right now; 0 at terminal states."""
    phi = -bp_scale(q_star, m0) * np.asarray(q_rem, float) * (np.asarray(spread_ticks) * tick / 2 + fee_take)
    return np.where(np.asarray(terminal, bool), 0.0, phi)


def shaped(r, phi_s, phi_s2):
    """r~ = r + Phi(s') - Phi(s)."""
    return np.asarray(r) + np.asarray(phi_s2) - np.asarray(phi_s)


def macro_reward_bp(fill_v, fill_p, m_t, v_target, v_left, p_after, d: int = 1) -> float:
    """Bandit reward for one macro action (RL doc 6.1), relative to the DECISION-time mid m_t:
    r = -1e4 / (v_target m_t) * [ sum_j v_j (p_j - m_t) + v_left (p_after - m_t) ]."""
    c = np.sum(fill_cost(fill_v, fill_p, m_t, d)) + fill_cost(v_left, p_after, m_t, d)
    return float(-1e4 / (v_target * m_t) * c)


def spread_timing_split(p, m_at_fill, m0, d: int = 1):
    """Cost decomposition d(p - m0) = d(p - m_t) [spread term] + d(m_t - m0) [timing term].
    The timing term is what OBI / OFI signals predict; report both in evaluation."""
    p, m_at_fill = np.asarray(p, float), np.asarray(m_at_fill, float)
    return d * (p - m_at_fill), d * (m_at_fill - m0)

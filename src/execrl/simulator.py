"""A vectorised TOY execution-slice environment, plus generators of logged data.

THIS IS A TEST FIXTURE, NOT A MARKET MODEL. It exists so every algorithm in the package can be run,
unit-tested and compared against a known ground truth on a laptop. Replace it with a calibrated
simulator (RL doc 10: queue-reactive + Hawkes order flow + propagator impact) before drawing any
conclusion about real execution. Keep the same API (reset/step/obs/mask) and everything else works.

Dynamics per decision step, buy side, all envs in lock-step (one slice = n_steps decisions):
 1. agent acts: TAKE buys min(rem, take_lots) at the ask (+taker fee); JOIN/IMPROVE/POST_DEEP place a
    resting buy at bid / bid+1 tick / bid-1 tick for the whole remaining quantity; CANCEL removes it;
 2. the mid moves -1 / 0 / +1 tick with probabilities driven by a latent AR(1) drift z:
        p_up = 0.25 + 0.15 tanh(z),   p_down = 0.25 - 0.15 tanh(z)
    the agent sees noisy versions of p_up and p_down (the "supervised signals" of RL doc 2.2);
 3. a resting order fills (up to fill_lots, at its price minus the maker rebate) with a probability
    that depends on how far it is from the PRE-move bid and on the move: fills are likely when the
    price falls (adverse selection is built in) and rare when it rises;
 4. the spread is redrawn (2 ticks with prob p_wide_spread, else 1); z evolves;
 5. after the last step the remainder is bought at the ask + clean-up premium (RL doc 2.4).

Observation (OBS_NAMES): remaining fraction, time left, position in the progress band, wide-spread
flag, the two noisy signals, has-order flag, distance of the resting order below the bid (ticks).
"""
from __future__ import annotations

import numpy as np

from .actions import Action, DEFAULT_MACROS, N_ACTIONS, action_mask, macro_mask
from .config import SliceConfig
from .data import BanditLog, Transitions
from .reward import potential

OBS_NAMES = ["rem_frac", "time_left", "band_pos", "wide_spread", "sig_up", "sig_dn", "has_order", "order_dist"]
OBS_DIM = len(OBS_NAMES)

# fill probability of a resting buy, rows: distance below the pre-move bid (-1 improve, 0 join, 1 deep),
# columns: mid move (down, flat, up). Orders 2+ ticks below the bid never fill.
FILL_P = np.array([[0.90, 0.45, 0.15],
                   [0.80, 0.25, 0.05],
                   [0.50, 0.00, 0.00]])


class SliceEnv:
    """n_envs slices simulated in parallel with numpy. reset() -> (obs, mask);
    step(actions) -> (obs, reward, done, mask, info). `done` is True for all envs after n_steps."""

    def __init__(self, cfg: SliceConfig, n_envs: int, seed: int = 0):
        self.cfg, self.n = cfg, n_envs
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ state helpers
    def _signals(self):
        th = np.tanh(self.z)
        self.p_up, self.p_dn = 0.25 + 0.15 * th, 0.25 - 0.15 * th
        e = self.rng.normal(0, self.cfg.signal_noise, (2, self.n))
        self.sig_up = np.clip(self.p_up + e[0], 0, 1)
        self.sig_dn = np.clip(self.p_dn + e[1], 0, 1)

    def _band(self):
        c = self.cfg
        target = self.t / c.n_steps
        return 1.0 - self.rem / c.q_star, target - c.band_width, target + c.band_width

    @property
    def bid(self):
        return self.mid - self.spread * self.cfg.tick / 2

    @property
    def ask(self):
        return self.mid + self.spread * self.cfg.tick / 2

    def obs(self) -> np.ndarray:
        c = self.cfg
        prog, lo, hi = self._band()
        has = ~np.isnan(self.order_px)
        dist = np.where(has, np.rint((self.bid - np.nan_to_num(self.order_px)) / c.tick), 0.0)
        return np.column_stack([self.rem / c.q_star, np.full(self.n, 1 - self.t / c.n_steps),
                                (prog - lo) / (hi - lo), (self.spread > 1).astype(float),
                                self.sig_up, self.sig_dn, has.astype(float), np.clip(dist, -1, 3)])

    def mask(self) -> np.ndarray:
        prog, lo, hi = self._band()
        return action_mask(prog, lo, hi, self.spread, ~np.isnan(self.order_px), self.rem <= 1e-12,
                           self.cfg.band_eps)

    def potential(self) -> np.ndarray:
        c = self.cfg
        return potential(self.rem, c.q_star, self.m0, self.spread, c.tick, c.fee_take_ticks * c.tick,
                         np.full(self.n, self.t >= c.n_steps))

    # ------------------------------------------------------------------ API
    def reset(self):
        c, n = self.cfg, self.n
        self.t = 0
        self.rem = np.full(n, c.q_star)
        self.mid = np.full(n, c.mid0)
        self.m0 = self.mid.copy()
        self.spread = np.where(self.rng.random(n) < c.p_wide_spread, 2, 1)
        self.z = self.rng.normal(size=n)
        self.order_px = np.full(n, np.nan)
        self._signals()
        return self.obs(), self.mask()

    def step(self, a):
        c, n, tick = self.cfg, self.n, self.cfg.tick
        a = np.asarray(a)
        active = self.rem > 1e-12
        phi_s = self.potential() if c.shaping else None
        fill_v = np.zeros(n)
        fill_vp = np.zeros(n)                       # sum v * p this step (for macro accounting)

        # 1. actions
        take = active & (a == Action.TAKE)
        v = np.where(take, np.minimum(self.rem, c.take_lots), 0.0)
        p = self.ask + c.fee_take_ticks * tick
        fill_v += v; fill_vp += v * p; self.rem -= v
        self.order_px[take | (a == Action.CANCEL)] = np.nan
        bid0 = self.bid
        for act, off in ((Action.JOIN, 0.0), (Action.IMPROVE, 1.0), (Action.POST_DEEP, -1.0)):
            sel = active & (a == act)
            off_ = np.where((act == Action.IMPROVE) & (self.spread < 2), 0.0, off)   # IMPROVE needs room
            self.order_px[sel] = (bid0 + off_ * tick)[sel]

        # 2. market move
        u = self.rng.random(n)
        move = np.where(u < self.p_up, 1, np.where(u < self.p_up + self.p_dn, -1, 0))

        # 3. passive fills against the pre-move bid
        has = ~np.isnan(self.order_px) & (self.rem > 1e-12)
        dist = np.rint((bid0 - np.nan_to_num(self.order_px)) / tick).astype(int)
        ok = has & (dist >= -1) & (dist <= 1)
        prob = np.zeros(n)
        prob[ok] = FILL_P[dist[ok] + 1, move[ok] + 1]
        filled = self.rng.random(n) < prob
        v = np.where(filled, np.minimum(self.rem, c.fill_lots), 0.0)
        p = np.nan_to_num(self.order_px) + c.fee_make_ticks * tick
        fill_v += v; fill_vp += v * p; self.rem -= v
        self.order_px[self.rem <= 1e-12] = np.nan

        # 4. state update
        self.mid = self.mid + move * tick
        self.spread = np.where(self.rng.random(n) < c.p_wide_spread, 2, 1)
        self.z = c.drift_ar * self.z + np.sqrt(1 - c.drift_ar ** 2) * self.rng.normal(size=n)
        self._signals()
        self.t += 1

        # 5. clean-up at the end of the slice
        done = self.t >= c.n_steps
        if done:
            v = self.rem.copy()
            p = self.ask + (c.fee_take_ticks + c.clean_extra_ticks) * tick
            fill_v += v; fill_vp += v * p; self.rem[:] = 0.0
            self.order_px[:] = np.nan

        cost = fill_vp - fill_v * self.m0          # sum v (p - m0), buy side
        r = -1e4 / (c.q_star * self.m0) * cost
        if c.shaping:
            r = r + self.potential() - phi_s
        info = {"fill_v": fill_v, "fill_vp": fill_vp, "cost_bp": -r if not c.shaping else None}
        return self.obs(), r, np.full(n, done), self.mask(), info


# ---------------------------------------------------------------------- data generators

def rollout(cfg: SliceConfig, policy, n_episodes: int, seed: int = 0, batch: int = 2048) -> Transitions:
    """Run `policy(obs, mask, rng) -> (actions, propensities)` for n_episodes slices and return
    every transition. Episode ids are the group ids (cluster standard errors by them)."""
    rng = np.random.default_rng(seed + 7)
    parts, ep0 = [], 0
    while ep0 < n_episodes:
        m = min(batch, n_episodes - ep0)
        env = SliceEnv(cfg, m, seed=seed + ep0)
        obs, mask = env.reset()
        for t in range(cfg.n_steps):
            a, p = policy(obs, mask, rng)
            obs2, r, done, mask2, _ = env.step(a)
            parts.append(dict(S=obs, A=a, R=r, S2=obs2, done=done.astype(float), mask=mask, mask2=mask2,
                              prop=p, group=np.arange(ep0, ep0 + m), step=np.full(m, t)))
            obs, mask = obs2, mask2
        ep0 += m
    cat = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
    return Transitions(**cat)


def episode_returns(tr: Transitions) -> np.ndarray:
    """Sum of rewards per episode (= -slice cost in bp when shaping is off)."""
    return np.bincount(tr.group, weights=tr.R)


def macro_bandit_log(cfg: SliceConfig, policy, n: int, seed: int = 0, macros=DEFAULT_MACROS) -> BanditLog:
    """Logged macro-action decisions (RL doc 6.1).

    For each start step k in 0..N-1-max_hold, run fresh envs for k steps with a random allowed
    atomic action (to diversify states), record the context x and the macro mask, let
    `policy(x, macro_mask, rng) -> (arms, propensities)` choose a macro, execute it (the macro's
    atomic action, then HOLD for hold_steps - 1 steps), and price the leftover at the ask plus the
    clean-up premium. Reward = macro_reward_bp relative to the decision-time mid."""
    rng = np.random.default_rng(seed + 11)
    max_hold = max(m.hold_steps for m in macros)
    starts = list(range(0, cfg.n_steps - max_hold))
    per = int(np.ceil(n / len(starts)))
    X, A, P, R, M, G = [], [], [], [], [], []
    gid = 0
    for k in starts:
        env = SliceEnv(cfg, per, seed=seed + 1000 * (k + 1))
        obs, mask = env.reset()
        for _ in range(k):
            pr = mask / mask.sum(1, keepdims=True)
            a = (pr.cumsum(1) > rng.random((per, 1))).argmax(1)
            obs, _, _, mask, _ = env.step(a)
        live = env.rem > 1e-12
        prog, lo, _ = env._band()
        mm = macro_mask(env.spread, prog < lo + cfg.band_eps, macros)
        arm, prop = policy(obs, mm, rng)
        m_t, v_target = env.mid.copy(), env.rem.copy()
        hold = np.array([macros[j].hold_steps for j in arm])
        kind = np.array([int(macros[j].kind) for j in arm])
        vp, vv = np.zeros(per), np.zeros(per)                # sum v*p and sum v of the macro's fills
        v_left, p_after = np.zeros(per), np.zeros(per)
        for h in range(max_hold):
            act = np.where(h == 0, kind, int(Action.HOLD))
            act = np.where(h == hold, int(Action.CANCEL), act)   # macro over: pull the order
            _, _, _, _, info = env.step(act)
            run = h < hold                                    # count fills only while the macro runs
            vp += np.where(run, info["fill_vp"], 0.0)
            vv += np.where(run, info["fill_v"], 0.0)
            end = h == hold - 1                               # leftover priced when the macro ends
            v_left[end] = (v_target - vv)[end]
            p_after[end] = (env.ask + (cfg.fee_take_ticks + cfg.clean_extra_ticks) * cfg.tick)[end]
        cost = (vp - vv * m_t) + v_left * (p_after - m_t)
        r = -1e4 / (np.maximum(v_target, 1e-12) * m_t) * cost
        X.append(obs[live]); A.append(arm[live]); P.append(prop[live]); R.append(r[live])
        M.append(mm[live]); G.append(np.arange(gid, gid + live.sum())); gid += live.sum()
    cat = lambda L: np.concatenate(L)[:n]
    return BanditLog(X=cat(X), A=cat(A), P=cat(P), R=cat(R), mask=cat(M), group=cat(G))

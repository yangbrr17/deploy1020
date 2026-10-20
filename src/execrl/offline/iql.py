"""Discrete implicit Q-learning (RL doc 7.2; Kostrikov, Nair, Levine 2022). PyTorch, CPU.

Three networks on the state s (all 2-layer MLPs):
    Q1, Q2 : s -> R^K     twin critics; the target uses min(Q1', Q2') (target nets, Polyak averaged)
    V      : s -> R        fitted by expectile regression, so it never queries unseen actions
    pi     : s -> logits   advantage-weighted behaviour cloning, masked softmax

    L_V  = E[ |kappa - 1(u < 0)| u^2 ],          u = min Q'(s, a) - V(s)          (a from the logs)
    L_Q  = E[ (r + (1 - done) V(s') - Q_k(s, a))^2 ]  for k = 1, 2
    L_pi = -E[ min(exp(beta (min Q'(s, a) - V(s))), adv_clip) * log pi(a | s) ]

kappa -> 0.5 gives the mean (SARSA-like); kappa -> 1 approaches max over logged actions.
beta -> 0 is behaviour cloning; larger beta is greedier. CPU budget: 1e6 transitions, 2e5 steps,
batch 512, 256 hidden units: about 1 hour on 8 P-cores (torch.set_num_threads(8)).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..config import IQLConfig


def mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class DiscreteIQL:
    def __init__(self, obs_dim: int, K: int, cfg: IQLConfig = IQLConfig()):
        torch.manual_seed(cfg.seed)
        self.cfg, self.K = cfg, K
        self.q1, self.q2 = mlp(obs_dim, K, cfg.hidden), mlp(obs_dim, K, cfg.hidden)
        self.q1t, self.q2t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.v, self.pi = mlp(obs_dim, 1, cfg.hidden), mlp(obs_dim, K, cfg.hidden)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=cfg.lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr)

    def fit(self, tr, log_every: int = 0):
        """tr: Transitions. Returns a list of (step, L_V, L_Q, L_pi)."""
        c = self.cfg
        T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt)
        S, S2, A, R, D = T(tr.S), T(tr.S2), T(tr.A, torch.long), T(tr.R), T(tr.done)
        M = T(tr.mask, torch.bool)
        rng = np.random.default_rng(c.seed)
        hist = []
        for step in range(c.steps):
            i = torch.as_tensor(rng.integers(0, len(S), c.batch_size))
            s, s2, a, r, d, m = S[i], S2[i], A[i], R[i], D[i], M[i]
            with torch.no_grad():
                qt = torch.min(self.q1t(s), self.q2t(s)).gather(1, a[:, None]).squeeze(1)
            # V: expectile regression towards the target Q of logged actions
            u = qt - self.v(s).squeeze(1)
            lv = (torch.abs(c.expectile - (u < 0).float()) * u ** 2).mean()
            self.opt_v.zero_grad(); lv.backward(); self.opt_v.step()
            # Q: one-step TD towards r + V(s')
            with torch.no_grad():
                y = r + (1 - d) * self.v(s2).squeeze(1)
            lq = sum(((q(s).gather(1, a[:, None]).squeeze(1) - y) ** 2).mean() for q in (self.q1, self.q2))
            self.opt_q.zero_grad(); lq.backward(); self.opt_q.step()
            # pi: advantage-weighted regression with the action mask
            with torch.no_grad():
                w = torch.clamp(torch.exp(c.beta * (qt - self.v(s).squeeze(1))), max=c.adv_clip)
            logits = self.pi(s).masked_fill(~m, -1e9)
            lp = -(w * torch.log_softmax(logits, 1).gather(1, a[:, None]).squeeze(1)).mean()
            self.opt_pi.zero_grad(); lp.backward(); self.opt_pi.step()
            # Polyak update of the target critics
            with torch.no_grad():
                for net, tgt in ((self.q1, self.q1t), (self.q2, self.q2t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - c.tau_target).add_(c.tau_target * p)
            if log_every and step % log_every == 0:
                hist.append((step, lv.item(), lq.item(), lp.item()))
        return hist

    @torch.no_grad()
    def act(self, S, mask) -> np.ndarray:
        logits = self.pi(torch.as_tensor(S, dtype=torch.float32))
        logits = logits.masked_fill(~torch.as_tensor(mask, dtype=torch.bool), -1e9)
        return logits.argmax(1).numpy()

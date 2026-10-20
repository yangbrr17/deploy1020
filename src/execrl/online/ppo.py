"""Masked PPO on the vectorised SliceEnv (RL doc 8.1-8.2). Research / validation only.

    rho_t = pi(a_t|s_t) / pi_old(a_t|s_t)
    L_clip = -E[ min(rho_t A_t, clip(rho_t, 1 - eps, 1 + eps) A_t) ]
    A_t (GAE) = sum_l (gamma lambda)^l delta_{t+l},   delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
    L = L_clip + c_v E[(V(s_t) - R_t)^2] - c_e E[H(pi(.|s_t))]
Masking: logits of disallowed actions are set to -1e9 before the softmax, both when acting and when
recomputing log-probabilities in the update (the mask is stored with each sample).

Why numpy-vectorised envs instead of subprocesses: the toy env steps 256 slices in microseconds, so
process overhead would dominate. A slow, realistic simulator should instead run in a process pool
(one env per worker, workers = physical cores, OMP_NUM_THREADS=1 per worker) and feed the same
`collect` interface.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..config import PPOConfig, SliceConfig
from ..simulator import OBS_DIM, SliceEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, K: int, hidden: int = 64):
        super().__init__()
        def body():
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor, self.critic = nn.Sequential(body(), nn.Linear(hidden, K)), nn.Sequential(body(), nn.Linear(hidden, 1))

    def dist(self, s, mask):
        logits = self.actor(s).masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def value(self, s):
        return self.critic(s).squeeze(-1)


def gae(rewards, values, dones, gamma: float, lam: float):
    """rewards, dones: (T, B); values: (T + 1, B) with values[T] the bootstrap (0 at the end of a slice).
    Returns advantages (T, B) and returns = advantages + values[:T]."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1])
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values[:T]


@torch.no_grad()
def collect(model, env: SliceEnv, T: int, rng):
    """One batch of complete slices (all envs start together and end after T steps)."""
    obs, mask = env.reset()
    buf = {k: [] for k in ("s", "m", "a", "logp", "v", "r", "d")}
    for _ in range(T):
        s, m = torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(mask)
        dist = model.dist(s, m)
        a = dist.sample()
        obs2, r, done, mask2, _ = env.step(a.numpy())
        for k, v in zip(buf, (obs, mask, a.numpy(), dist.log_prob(a).numpy(), model.value(s).numpy(), r, done.astype(float))):
            buf[k].append(v)
        obs, mask = obs2, mask2
    out = {k: np.stack(v) for k, v in buf.items()}          # (T, B, ...)
    out["v"] = np.concatenate([out["v"], np.zeros((1, env.n))])   # terminal bootstrap = 0
    return out


def train_ppo(slice_cfg: SliceConfig, cfg: PPOConfig = PPOConfig(), K: int = 6, log=print):
    """Returns (model, history). history[i] = mean slice return (bp) of iteration i's batch."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = ActorCritic(OBS_DIM, K, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1 - i / max(cfg.iterations, 1))
    env = SliceEnv(slice_cfg, cfg.n_envs, seed=cfg.seed)
    T = slice_cfg.n_steps
    hist = []
    for it in range(cfg.iterations):
        b = collect(model, env, T, rng)
        adv, ret = gae(b["r"], b["v"], b["d"], cfg.gamma, cfg.gae_lambda)
        hist.append(float(b["r"].sum(0).mean()))
        flat = lambda x: torch.as_tensor(x.reshape(T * env.n, *x.shape[2:]))
        s, m, a = flat(b["s"]).float(), flat(b["m"]).bool(), flat(b["a"]).long()
        old_logp, A, R = flat(b["logp"]).float(), flat(adv).float(), flat(ret).float()
        A = (A - A.mean()) / (A.std() + 1e-8)
        n = len(s)
        for _ in range(cfg.epochs):
            perm = torch.as_tensor(rng.permutation(n))
            for j in range(0, n, cfg.minibatch):
                i = perm[j:j + cfg.minibatch]
                dist = model.dist(s[i], m[i])
                rho = torch.exp(dist.log_prob(a[i]) - old_logp[i])
                l_clip = -torch.min(rho * A[i], torch.clamp(rho, 1 - cfg.clip, 1 + cfg.clip) * A[i]).mean()
                l_v = ((model.value(s[i]) - R[i]) ** 2).mean()
                loss = l_clip + cfg.vf_coef * l_v - cfg.ent_coef * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
        sched.step()
        if log and (it % max(1, cfg.iterations // 10) == 0 or it == cfg.iterations - 1):
            log(f"PPO iter {it + 1}/{cfg.iterations}: mean slice return {hist[-1]:.3f} bp")
    return model, hist


def ppo_policy(model):
    """Wrap a trained model as a package policy (greedy; propensity 1)."""
    @torch.no_grad()
    def policy(x, mask, rng):
        d = model.dist(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(mask))
        return d.probs.argmax(1).numpy(), np.ones(len(x))
    return policy


def distill_tree(policy, slice_cfg: SliceConfig, depth: int = 4, n_episodes: int = 4000, seed: int = 0):
    """Fit a depth-limited decision tree to the policy's actions on its own state distribution
    (RL doc 8.2). Returns (tree, wrapped tree policy). Deploy the tree only if it is not worse."""
    from sklearn.tree import DecisionTreeClassifier
    from ..simulator import rollout
    tr = rollout(slice_cfg, policy, n_episodes, seed=seed)
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=200, random_state=seed).fit(tr.S, tr.A)

    def tree_policy(x, mask, rng):
        proba = np.zeros((len(x), mask.shape[1]))
        proba[:, tree.classes_] = tree.predict_proba(x)
        return np.where(mask, proba, -1).argmax(1), np.ones(len(x))
    return tree, tree_policy

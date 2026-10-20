"""Unit and smoke tests for execrl. Small sizes so the whole file runs in about a minute on a laptop.

    PYTHONPATH=src python -m pytest -q src/tests/test_execrl.py
"""
import numpy as np
import pytest

from execrl import baselines, reward
from execrl import logging_policy as lp
from execrl.actions import Action, action_mask
from execrl.bandit import ope, policy_tree
from execrl.config import ExperimentConfig, SliceConfig
from execrl.simulator import SliceEnv, episode_returns, rollout


def small_cfg():
    cfg = ExperimentConfig()
    cfg.bandit.n_samples = 8_000
    cfg.bandit.n_folds = 3
    cfg.bandit.lgbm.n_estimators = 80
    cfg.fqi.n_episodes = 1_500
    cfg.fqi.lgbm.n_estimators = 60
    cfg.iql.steps = 300
    cfg.ppo.iterations = 3
    cfg.ppo.n_envs = 64
    return cfg


def test_mask_rules():
    m = action_mask(progress=[0.0, 0.9, 0.5], lower=[0.3, 0.3, 0.3], upper=[0.7, 0.7, 0.7],
                    spread_ticks=[1, 2, 1], has_order=[False, True, False], done=[False, False, False],
                    eps_band=0.05)
    assert m[0].tolist() == [True, False, False, False, False, False]          # behind, 1-tick spread: TAKE only
    assert not m[1, Action.TAKE] and m[1, Action.CANCEL]                        # ahead: no TAKE
    assert not m[2, Action.IMPROVE] and not m[2, Action.CANCEL]


def test_rewards_sum_to_minus_cost_and_shaping_is_invariant():
    """sum_n r_n = -C, and with shaping sum_n r~_n = -C - Phi(s_0)."""
    for shaping in (False, True):
        cfg = SliceConfig(shaping=shaping)
        env = SliceEnv(cfg, 500, seed=3)
        obs, mask = env.reset()
        phi0 = env.potential() if shaping else 0.0
        total, vp, vv = np.zeros(500), np.zeros(500), np.zeros(500)
        rng = np.random.default_rng(0)
        for _ in range(cfg.n_steps):
            a, _ = lp.uniform(obs, mask, rng)
            obs, r, done, mask, info = env.step(a)
            total += r; vp += info["fill_vp"]; vv += info["fill_v"]
        cost = reward.bp_scale(cfg.q_star, cfg.mid0) * (vp - vv * cfg.mid0)
        np.testing.assert_allclose(vv, cfg.q_star)                              # everything bought
        np.testing.assert_allclose(total, -cost - phi0, atol=1e-9)


def test_logging_propensities_are_consistent():
    rng = np.random.default_rng(0)
    env = SliceEnv(SliceConfig(), 2000, seed=1)
    obs, mask = env.reset()
    pol = lp.epsilon_rule(baselines.slice_rule, 0.1)
    a, p = pol(obs, mask, rng)
    assert mask[np.arange(len(a)), a].all() and (p > 0).all() and (p <= 1).all()
    rule = baselines.slice_rule(obs, mask)
    frac_rule = np.mean(a == rule)
    assert abs(frac_rule - np.mean(p[a == rule].mean())) < 0.05                # empirical ~ recorded


def test_dr_is_unbiased_on_synthetic_bandit():
    """Known reward function, uniform logging: DR of a fixed policy within 3 se of the truth."""
    rng = np.random.default_rng(0)
    n, K = 20_000, 3
    X = rng.normal(size=(n, 2))
    true_mu = np.column_stack([X[:, 0], -X[:, 0], 0.3 * np.ones(n)])
    a = rng.integers(0, K, n)
    r = true_mu[np.arange(n), a] + rng.normal(size=n)
    p = np.full(n, 1 / K)
    pi = ope.one_hot(np.where(X[:, 0] > 0, 0, 1), K)                          # arm 0 if x > 0 else arm 1
    mu_bad = np.zeros((n, K))                                                   # deliberately wrong model
    est = ope.evaluate(pi, a, r, p, mu_bad, np.arange(n), clip=1e9)
    truth = np.abs(X[:, 0]).mean()
    assert abs(est["dr"] - truth) < 3 * est["dr_se"]


def test_policy_tree_recovers_a_split():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(5000, 3))
    G = np.column_stack([np.where(X[:, 1] > 0.6, 1.0, -1.0), np.zeros(5000)]) + rng.normal(scale=0.1, size=(5000, 2))
    t = policy_tree.PolicyTree(depth=1, n_grid=19).fit(X, G)
    assert t.tree["j"] == 1 and abs(t.tree["c"] - 0.6) < 0.06


def test_train_bandit_smoke(tmp_path):
    from execrl.train import train_bandit
    out = train_bandit(small_cfg(), out_dir=str(tmp_path), log=lambda *a: None)
    m = out["metrics"]
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "rules.txt").exists()
    assert m["truth"]["greedy"] > m["truth"]["logging"]                       # learning helps in the toy env


def test_train_fqi_smoke(tmp_path):
    from execrl.train import train_fqi
    m = train_fqi(small_cfg(), out_dir=str(tmp_path), log=lambda *a: None)["metrics"]
    assert np.isfinite(m["fqe"]["fqi"]) and "truth" in m


def test_train_iql_and_ppo_smoke(tmp_path):
    pytest.importorskip("torch")
    from execrl.train import train_iql, train_ppo_run
    cfg = small_cfg()
    assert np.isfinite(train_iql(cfg, out_dir=str(tmp_path / "iql"), log=lambda *a: None)["metrics"]["truth"]["iql"])
    assert len(train_ppo_run(cfg, out_dir=str(tmp_path / "ppo"), log=lambda *a: None)["metrics"]["history"]) == 3


def test_config_roundtrip(tmp_path):
    cfg = ExperimentConfig()
    cfg.bandit.lgbm.num_leaves = 15
    path = tmp_path / "c.json"
    cfg.to_json(path)
    back = ExperimentConfig.from_json(str(path))
    assert back.bandit.lgbm.num_leaves == 15
    partial = ExperimentConfig.from_json('{"fqi": {"support_tau": 0.05}}')
    assert partial.fqi.support_tau == 0.05 and partial.slice.n_steps == SliceConfig().n_steps

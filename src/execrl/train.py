"""Training entry points and the command line.

Each train_* function follows the same five steps, so adding a method means copying one of them:
    1. data      : take a BanditLog / Transitions, or generate one in the toy simulator
    2. split     : by group (parent order), never by row
    3. fit       : the method's own module (bandit/, offline/, online/)
    4. evaluate  : off-policy estimate on the held-out split (DR or FQE) + simulator ground truth
                   (ground truth exists only for simulated data; on real logs it is the shadow run)
    5. save      : config.json, metrics.json, model files in out_dir
They return a dict {"metrics": ..., "policy": callable, ...}. The CLI just calls them.

    PYTHONPATH=src python -m execrl.train bandit --out runs/bandit
    PYTHONPATH=src python -m execrl.train fqi --out runs/fqi --threads 8
    PYTHONPATH=src python -m execrl.train ppo --out runs/ppo --iterations 100
    PYTHONPATH=src python -m execrl.train all --config my.json --out runs/all
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from . import baselines, compute
from . import logging_policy as lp
from .actions import ACTION_NAMES, DEFAULT_MACROS, N_ACTIONS
from .bandit import ope, policy_tree, reward_model
from .config import ExperimentConfig
from .data import BanditLog, Transitions, split_by_group
from .evaluate import simulate_value
from .simulator import OBS_NAMES, macro_bandit_log, rollout


def _save(out_dir, cfg: ExperimentConfig, metrics: dict, **objects):
    """Write config.json, metrics.json and each object (.txt as text, .pt via torch.save, else pickle)."""
    if not out_dir:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    for name, obj in objects.items():
        if name.endswith(".txt"):
            (out / name).write_text(obj)
        elif name.endswith(".pt"):
            import torch
            torch.save(obj, out / name)
        else:
            with open(out / name, "wb") as f:
                pickle.dump(obj, f)


def _setup(cfg: ExperimentConfig, log):
    """Resolve and apply the thread count (0 = number of performance cores)."""
    n = compute.resolve(cfg.compute.threads)
    compute.set_threads(n)
    log(f"threads={n}  cpu={compute.detect_cpu()}")


# ------------------------------------------------------------------------------------ bandit

def train_bandit(cfg: ExperimentConfig = ExperimentConfig(), data: BanditLog | None = None,
                 out_dir: str | None = None, log=print) -> dict:
    """Macro-action contextual bandit with cross-fitted DR evaluation and a policy tree (RL doc 6)."""
    _setup(cfg, log)
    t0 = time.perf_counter()
    K, bc = len(DEFAULT_MACROS), cfg.bandit
    logging = lp.epsilon_rule(baselines.macro_rule, cfg.logging.epsilon)
    simulated = data is None
    if simulated:                                                         # 1. data
        data = macro_bandit_log(cfg.slice, logging, bc.n_samples, seed=cfg.seed)
    tr_idx, te_idx = split_by_group(data.group, bc.test_frac, cfg.seed)   # 2. split
    tr, te = data.subset(tr_idx), data.subset(te_idx)

    mu_oof, models = reward_model.cross_fit(tr.X, tr.A, tr.R, tr.group, K, bc.n_folds, bc.lgbm, cfg.seed)  # 3. fit
    mu_te = reward_model.predict_ensemble(models, te.X, K)               # fitted without test rows

    def greedy_arms(X, mask):
        return np.where(mask, reward_model.predict_ensemble(models, X, K), -np.inf).argmax(1)

    pi_greedy = ope.one_hot(greedy_arms(te.X, te.mask), K)               # 4. evaluate
    metrics = {"n_train": len(tr), "n_test": len(te),
               "logging": ope.logging_policy_value(te.R, te.group),
               "greedy": ope.evaluate(pi_greedy, te.A, te.R, te.P, mu_te, te.group, bc.weight_clip),
               "greedy_gate": ope.gate(pi_greedy, te.A, te.R, te.P, mu_te, te.group, bc.weight_clip, bc.gate_z),
               "clip_sensitivity": ope.clip_sensitivity(pi_greedy, te.A, te.R, te.P, mu_te, te.group)}
    tree, rules = None, ""
    if bc.tree_depth > 0:
        G = policy_tree.dr_scores(mu_oof, tr.A, tr.R, tr.P, tr.mask, bc.weight_clip)
        tree = policy_tree.PolicyTree(bc.tree_depth, bc.tree_grid).fit(tr.X, G)
        rules = tree.rules(OBS_NAMES, [m.name for m in DEFAULT_MACROS])

        def tree_arms(X, mask):
            a = tree.predict(X)
            bad = ~mask[np.arange(len(a)), a]
            a[bad] = greedy_arms(X[bad], mask[bad]) if bad.any() else a[bad]
            return a
        pi_tree = ope.one_hot(tree_arms(te.X, te.mask), K)
        metrics["tree"] = ope.evaluate(pi_tree, te.A, te.R, te.P, mu_te, te.group, bc.weight_clip)
        metrics["tree_gate"] = ope.gate(pi_tree, te.A, te.R, te.P, mu_te, te.group, bc.weight_clip, bc.gate_z)
    if simulated:                                                         # ground truth (toy only)
        def as_policy(f):
            return lambda x, m, rng: (f(x, m), np.ones(len(x)))
        truth = lambda pol: float(macro_bandit_log(cfg.slice, pol, 20_000, seed=cfg.seed + 999).R.mean())
        metrics["truth"] = {"logging": truth(logging), "greedy": truth(as_policy(greedy_arms))}
        if tree is not None:
            metrics["truth"]["tree"] = truth(as_policy(tree_arms))
    metrics["seconds"] = time.perf_counter() - t0
    log(json.dumps({k: metrics[k] for k in ("logging", "greedy", "greedy_gate")}, default=float))
    if rules:
        log("policy tree:\n" + rules)
    _save(out_dir, cfg, metrics, **{"reward_models.pkl": models, "policy_tree.pkl": tree, "rules.txt": rules})
    return {"metrics": metrics, "models": models, "tree": tree, "policy": greedy_arms}


# ------------------------------------------------------------------------------------ offline RL

def _logged_transitions(cfg: ExperimentConfig, n_episodes: int):
    """Toy logs: the slice rule with epsilon exploration, propensities recorded."""
    logging = lp.epsilon_rule(baselines.slice_rule, cfg.logging.epsilon)
    return logging, rollout(cfg.slice, logging, n_episodes, seed=cfg.seed)


def _baseline_truth(cfg, logging):
    """Simulator value of the logging policy and the two deterministic rules (toy ground truth)."""
    return {"logging": simulate_value(logging, cfg.slice)[0],
            "slice_rule": simulate_value(lp.greedy_rule(baselines.slice_rule), cfg.slice)[0],
            "one_step_rule": simulate_value(lp.greedy_rule(baselines.one_step_rule()), cfg.slice)[0]}


def train_fqi(cfg: ExperimentConfig = ExperimentConfig(), data: Transitions | None = None,
              out_dir: str | None = None, log=print) -> dict:
    """Batch-constrained FQI with LightGBM, evaluated by FQE and (toy) simulation (RL doc 7.1, 7.4)."""
    from .offline.behaviour import BehaviourModel
    from .offline.fqe import fqe
    from .offline.fqi import constrained_fqi, constrained_greedy
    _setup(cfg, log)
    t0 = time.perf_counter()
    fc = cfg.fqi
    simulated = data is None
    logging = None
    if simulated:
        logging, data = _logged_transitions(cfg, fc.n_episodes)
    tr_idx, te_idx = split_by_group(data.group, fc.test_frac, cfg.seed)
    tr, te = data.subset(tr_idx), data.subset(te_idx)

    beh = BehaviourModel(N_ACTIONS, fc.lgbm).fit(tr.S, tr.A)
    n_iter = fc.n_iter or cfg.slice.n_steps
    q = constrained_fqi(tr.S, tr.A, tr.R, tr.S2, tr.done, tr.mask2, beh.predict_proba(tr.S2, tr.mask2),
                        N_ACTIONS, n_iter, fc.support_tau, fc.lgbm)

    def fqi_actions(S, mask):
        return constrained_greedy(q, S, mask, beh.predict_proba(S, mask), N_ACTIONS, fc.support_tau)

    metrics = {"n_train": len(tr), "n_test": len(te), "n_iter": n_iter,
               "fqe": {"fqi": fqe(te, fqi_actions, N_ACTIONS, n_iter, fc.lgbm),
                       "slice_rule": fqe(te, baselines.slice_rule, N_ACTIONS, n_iter, fc.lgbm)},
               "logged_mean_return": float(np.bincount(te.group, weights=te.R)[np.unique(te.group)].mean())}
    if simulated:
        pol = lambda x, m, rng: (fqi_actions(x, m), np.ones(len(x)))
        metrics["truth"] = {"fqi": simulate_value(pol, cfg.slice)[0], **_baseline_truth(cfg, logging)}
    metrics["seconds"] = time.perf_counter() - t0
    log(json.dumps(metrics, default=float))
    _save(out_dir, cfg, metrics, **{"q_model.pkl": q, "behaviour.pkl": beh})
    return {"metrics": metrics, "q": q, "behaviour": beh, "policy": fqi_actions}


def train_iql(cfg: ExperimentConfig = ExperimentConfig(), data: Transitions | None = None,
              out_dir: str | None = None, log=print) -> dict:
    """Discrete IQL on the same logged transitions (RL doc 7.2)."""
    from .offline.fqe import fqe
    from .offline.iql import DiscreteIQL
    _setup(cfg, log)
    t0 = time.perf_counter()
    simulated = data is None
    logging = None
    if simulated:
        logging, data = _logged_transitions(cfg, cfg.fqi.n_episodes)
    tr_idx, te_idx = split_by_group(data.group, cfg.fqi.test_frac, cfg.seed)
    tr, te = data.subset(tr_idx), data.subset(te_idx)
    agent = DiscreteIQL(tr.S.shape[1], N_ACTIONS, cfg.iql)
    hist = agent.fit(tr, log_every=max(cfg.iql.steps // 10, 1))
    metrics = {"n_train": len(tr), "loss_history": hist,
               "fqe": {"iql": fqe(te, agent.act, N_ACTIONS, cfg.slice.n_steps, cfg.fqi.lgbm)}}
    if simulated:
        pol = lambda x, m, rng: (agent.act(x, m), np.ones(len(x)))
        metrics["truth"] = {"iql": simulate_value(pol, cfg.slice)[0], **_baseline_truth(cfg, logging)}
    metrics["seconds"] = time.perf_counter() - t0
    log(json.dumps({k: v for k, v in metrics.items() if k != "loss_history"}, default=float))
    _save(out_dir, cfg, metrics, **{"iql_policy.pt": agent.pi.state_dict()})
    return {"metrics": metrics, "agent": agent, "policy": agent.act}


# ------------------------------------------------------------------------------------ PPO

def train_ppo_run(cfg: ExperimentConfig = ExperimentConfig(), out_dir: str | None = None, log=print,
                  tree_depth: int = 4) -> dict:
    """Masked PPO in the toy simulator, then distillation to a depth-4 tree (RL doc 8)."""
    from .online.ppo import distill_tree, ppo_policy, train_ppo
    _setup(cfg, log)
    t0 = time.perf_counter()
    model, hist = train_ppo(cfg.slice, cfg.ppo, N_ACTIONS, log)
    pol = ppo_policy(model)
    tree, tree_pol = distill_tree(pol, cfg.slice, tree_depth, seed=cfg.seed)
    logging = lp.epsilon_rule(baselines.slice_rule, cfg.logging.epsilon)
    metrics = {"history": hist,
               "truth": {"ppo": simulate_value(pol, cfg.slice)[0], "ppo_tree": simulate_value(tree_pol, cfg.slice)[0],
                         **_baseline_truth(cfg, logging)}}
    metrics["seconds"] = time.perf_counter() - t0
    log(json.dumps(metrics["truth"], default=float))
    from sklearn.tree import export_text
    rules = export_text(tree, feature_names=OBS_NAMES, class_names=[ACTION_NAMES[c] for c in tree.classes_])
    _save(out_dir, cfg, metrics, **{"ppo.pt": model.state_dict(), "ppo_tree.pkl": tree, "rules.txt": rules})
    return {"metrics": metrics, "model": model, "tree": tree, "policy": pol}


# ------------------------------------------------------------------------------------ CLI

def main(argv=None):
    """Command line: python -m execrl.train {bandit,fqi,iql,ppo,all} [--config ...] [--out ...]."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["bandit", "fqi", "iql", "ppo", "all"])
    ap.add_argument("--config", help="JSON file or string; missing keys keep defaults")
    ap.add_argument("--out", default=None, help="output directory (config, metrics, models)")
    ap.add_argument("--threads", type=int, default=None, help="override compute.threads (0 = auto)")
    ap.add_argument("--iterations", type=int, default=None, help="PPO iterations")
    ap.add_argument("--steps", type=int, default=None, help="IQL gradient steps")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = ExperimentConfig.from_json(a.config) if a.config else ExperimentConfig()
    if a.threads is not None: cfg.compute.threads = a.threads
    if a.iterations is not None: cfg.ppo.iterations = a.iterations
    if a.steps is not None: cfg.iql.steps = a.steps
    if a.seed is not None: cfg.seed = a.seed
    sub = lambda name: str(Path(a.out) / name) if a.out else None
    runs = {"bandit": lambda: train_bandit(cfg, out_dir=sub("bandit") if a.method == "all" else a.out),
            "fqi": lambda: train_fqi(cfg, out_dir=sub("fqi") if a.method == "all" else a.out),
            "iql": lambda: train_iql(cfg, out_dir=sub("iql") if a.method == "all" else a.out),
            "ppo": lambda: train_ppo_run(cfg, out_dir=sub("ppo") if a.method == "all" else a.out)}
    for name in (runs if a.method == "all" else [a.method]):
        print(f"=== {name} ===")
        runs[name]()


if __name__ == "__main__":
    main()

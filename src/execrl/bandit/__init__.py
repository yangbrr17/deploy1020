"""Contextual bandit over macro actions (RL doc 6): the recommended first step.

    reward_model.py   mu_hat(x, a) with LightGBM; cross-fitted out-of-fold predictions for DR
    ope.py            IPW / SNIPW / DR policy values with cluster standard errors; deployment gate
    policy_tree.py    depth-1/2 policy tree on DR scores (exhaustive search on a quantile grid)
    thompson.py       linear Thompson sampling inside the safe set, with Monte-Carlo propensities
"""

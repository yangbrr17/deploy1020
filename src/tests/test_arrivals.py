"""Checks that the arrival tests accept data generated under their own null and reject clear violations.

Run from the repository root:  PYTHONPATH=src python -m pytest -q src/tests
"""
import numpy as np

from arrivals import models, queue, report, schema, synth
from arrivals import stats as st


def test_exponential_null_not_rejected_and_weibull_rejected():
    rng = np.random.default_rng(0)
    x = rng.exponential(2.0, 20_000)
    assert st.dispersion_test(x)["p_value"] > 0.01
    assert st.ks_exponential(x, 99, rng)["p_value"] > 0.01
    assert st.ljung_box(x)["p_value"] > 0.01
    w = rng.weibull(0.7, 20_000)                      # more short and long gaps than exponential
    r = st.dispersion_test(w)
    assert r["p_value"] < 1e-6 and r["effect"] > 1.2


def test_count_dispersion_poisson():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 10_000, 20_000))       # homogeneous Poisson given N
    r = st.count_dispersion(t, 10_000, 10.0)
    assert abs(r["effect"] - 1) < 0.1


def test_hawkes_residuals_pass_on_hawkes_data():
    rng = np.random.default_rng(2)
    t = synth._hawkes_times(0.5, 0.8, 1.2, 4000.0, rng)
    fit = models.fit_hawkes_exp(t, 4000.0)
    assert abs(fit["branching"] - 0.8 / 1.2) < 0.1
    e = models.hawkes_residuals(fit["params"], t)
    assert st.ks_exponential(e, 99, rng)["p_value"] > 0.01


def test_cont_de_larrard_exact_case():
    """Poisson flows, no seasonality: the PIT must be uniform and phi calibrated."""
    tr, qu, s = synth.make_day(seed=1, day_len=23400, hawkes=(0.8, 0.0, 1.0), season=False)
    e2 = schema.mid_change_events(qu, s, 1000, np.random.default_rng(0))
    rates = queue.estimate_rates(qu, tr, s)
    fp = queue.first_passage_sample(e2, s.seconds, rates["bid"]["unit"], rates["ask"]["unit"])
    assert queue.uniform_ks(queue.pit_first_passage(fp, rates))["p_value"] > 0.01
    assert st.calibration_test(queue.phi_for_sample(fp), fp["up"])["p_value"] > 0.01


def test_phi_matches_known_values():
    assert abs(queue.up_probability(3, 3) - 0.5) < 1e-6
    assert abs(queue.up_probability(5, 2) - 0.756) < 2e-3


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert st.benjamini_hochberg(p, 0.05).sum() == 2


def test_report_end_to_end_small():
    days = synth.make_days(2, day_len=3000.0)
    frames = [("SYN", f"d{i}", tr, qu, s) for i, (tr, qu, s) in enumerate(days)]
    df = report.run_frames(frames, report.Config(resolution_ns=1000, n_boot=49))
    assert {"H1", "H2", "H3", "H4", "H5", "H6"} <= set(df["hypothesis"])
    assert df["verdict"].notna().all()

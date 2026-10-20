"""Arrival-time hypothesis tests for order-book event streams.

Package layout (each module has a docstring with the details):

    schema.py         input tables, validation, cleaning, event-series extraction, durations
    stats.py          generic test statistics (dispersion, KS with bootstrap, Ljung-Box, LR, Vuong, Hill, ...)
    distributions.py  marginal duration fits: exponential / Weibull / gamma / generalised gamma / log-normal
    models.py         seasonality (diurnal) adjustment, Weibull-ACD, exponential Hawkes, time-rescaling
    queue.py          Cont-de Larrard queue-depletion formulas and the empirical queue-flow estimates
    synth.py          synthetic data in the same schema, for a dry run before touching real data
    report.py         runs every hypothesis H1-H9 on one symbol-day and writes one tidy results table

Quick start (from the repository root):

    PYTHONPATH=src python -m arrivals.report --synthetic --out results.csv
    PYTHONPATH=src python -m arrivals.report --trades trades.csv --quotes quotes.csv --out results.csv

The documentation page is docs/signal-model/arrival-tests.md.
"""

__all__ = ["schema", "stats", "distributions", "models", "queue", "synth", "report"]

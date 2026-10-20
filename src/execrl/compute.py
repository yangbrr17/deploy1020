"""CPU detection and thread settings for a CPU-only box (12th-gen Intel i9, no GPU).

12th-gen i9 parts are HYBRID: performance cores (P, with hyper-threading) + efficiency cores (E, no HT).
    i9-12900K / KF / KS / 12900 / F / T     8P + 8E = 16 cores, 24 threads
    i9-12900HX / 12950HX (laptop)           8P + 8E = 16 cores, 24 threads
    i9-12900H / HK (laptop)                 6P + 8E = 14 cores, 20 threads
No 12th-gen i9 has 12 cores; check yours with `lscpu` (Linux) or Task Manager -> Performance (Windows).

Rules of thumb (measure with `python -m execrl.compute --bench` before trusting them):
- LightGBM / numpy BLAS / torch intra-op: threads = number of P-cores (8). Adding E-cores often
  gives little or makes it slower because work is split evenly and waits for the slow cores.
- Many independent jobs (Optuna trials, simulator workers, per-day tests): processes = physical
  cores (16) with 1 thread each; total threads must not exceed logical threads (24).
- Never nest: n_workers x threads_per_worker <= physical cores. Set OMP_NUM_THREADS=1 inside workers.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _cpulist_len(path: Path) -> int:
    """Count CPUs in a Linux cpulist file like '0-15' or '16-23'."""
    try:
        n = 0
        for part in path.read_text().strip().split(","):
            a, _, b = part.partition("-")
            n += (int(b) - int(a) + 1) if b else 1
        return n
    except (OSError, ValueError):
        return 0


def detect_cpu() -> dict:
    """{'logical', 'physical', 'p_threads', 'e_cores', 'hybrid'}. Linux exposes hybrid topology in
    /sys/devices/cpu_core (P-core threads) and /sys/devices/cpu_atom (E-cores). Elsewhere we fall back
    to os.cpu_count() and the env overrides EXECRL_P_CORES / EXECRL_PHYSICAL_CORES."""
    logical = os.cpu_count() or 1
    p_threads = _cpulist_len(Path("/sys/devices/cpu_core/cpus"))
    e_cores = _cpulist_len(Path("/sys/devices/cpu_atom/cpus"))
    hybrid = p_threads > 0 and e_cores > 0
    physical = None
    try:  # count unique (physical id, core id) pairs
        ids, phys, core = set(), None, None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
                ids.add((phys, core))
        physical = len(ids) or None
    except OSError:
        pass
    physical = int(os.environ.get("EXECRL_PHYSICAL_CORES", physical or logical))
    p_cores = int(os.environ.get("EXECRL_P_CORES", p_threads // 2 if hybrid else physical))
    return {"logical": logical, "physical": physical, "p_cores": p_cores, "e_cores": e_cores,
            "hybrid": hybrid}


def recommended_threads(kind: str = "train") -> int:
    """Threads for ONE process. kind: 'train' (LightGBM / torch training), 'infer' (latency: 1-2
    threads, a single decision is tiny), 'worker' (inside a process pool: 1)."""
    info = detect_cpu()
    return {"train": max(1, info["p_cores"]), "infer": 1, "worker": 1}.get(kind, info["p_cores"])


def recommended_workers() -> int:
    """Processes for embarrassingly parallel jobs: physical cores (P + E)."""
    return max(1, detect_cpu()["physical"])


def set_threads(n: int) -> None:
    """Set BLAS/OpenMP env vars and torch's intra-op threads. Call BEFORE heavy imports when possible;
    env vars only affect libraries that have not started their thread pools yet."""
    for v in THREAD_VARS:
        os.environ[v] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass


def resolve(threads: int) -> int:
    """0 -> recommended_threads('train')."""
    return threads if threads > 0 else recommended_threads("train")


def bench_lightgbm(n: int = 200_000, d: int = 30, thread_list=(1, 4, 8, 16, 24)) -> dict:
    """Seconds to fit a 200-tree LightGBM for several thread counts; pick the fastest for your box."""
    import numpy as np
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)); y = X[:, 0] - X[:, 1] ** 2 + rng.normal(size=n)
    out = {}
    for t in thread_list:
        if t > (os.cpu_count() or 1):
            continue
        t0 = time.perf_counter()
        lgb.LGBMRegressor(n_estimators=200, num_threads=t, verbose=-1).fit(X, y)
        out[t] = round(time.perf_counter() - t0, 2)
    return out


if __name__ == "__main__":
    import sys
    print(detect_cpu())
    print("train threads:", recommended_threads("train"), " workers:", recommended_workers())
    if "--bench" in sys.argv:
        print("lightgbm seconds by threads:", bench_lightgbm())

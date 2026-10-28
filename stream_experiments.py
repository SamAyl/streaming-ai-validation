"""Robustness and validation experiments for the streaming-AI manuscript (Section 5 additions).

A  Forecast timing/amplitude error, Poisson arrivals: fixed, reactive, forecast, forecast+reactive (2,000 trials).
B  Checkpoint interval: exact renewal model under exponential failures vs. the first-order J(I) and the
   Young and Daly optima, checked by Monte Carlo.
C  Event-time watermarks: bounded-out-of-orderness watermark lag vs. result latency, completeness,
   late events, dropped events and decision revisions on a synthetic out-of-order stream.
Run: python stream_experiments.py -> results.json (+ figure via stream_figure.py)
"""
import json, math
import numpy as np

OUT = {}
rng = np.random.default_rng(2026)

# ===================================================================== A  forecast robustness
MU, BASE, H = 5000.0, 12000.0, 600

def run_schedule(arr, workers):
    q = 0.0; peak = 0.0; expo = 0.0; drain = None; ws = 0.0
    for t in range(H):
        c = workers[t] * MU
        q = max(0.0, q + arr[t] - c)
        peak = max(peak, q); expo += q; ws += workers[t]
    return peak, expo, ws

def det_grid():
    rows = []
    arr = np.full(H, BASE); arr[180:240] = 30000.0
    for eps in (-30, -15, 0, 10, 20, 30):
        for nf in (6, 7):
            w = np.full(H, 3); s = 180 + eps; w[max(0, s):s + 80] = nf
            pk, ex, ws = run_schedule(arr, w)
            rows.append(dict(timing_error_s=eps, forecast_workers=nf, peak=pk, exposure=ex, worker_s=ws))
    return rows
OUT["A_grid"] = det_grid()

def policies(arr, onset, dur, amp_pred, eps, react_delay=30, cool=60, hyb=False):
    out = {}
    # fixed
    out["fixed"] = np.full(H, 3)
    # reactive: detects queue growth at onset, adds capacity sized to observed rate after delay, holds until empty + cooldown
    def reactive(base_w):
        w = base_w.copy(); q = 0.0; active = False; t_on = None; t_empty = None
        for t in range(H):
            c = w[t] * MU; q = max(0.0, q + arr[t] - c)
            if not active and q > 0 and t_on is None:
                t_on = t + react_delay
                need = int(math.ceil(arr[t] / MU))
            if t_on is not None and t >= t_on and not active:
                active = True
            if active:
                w[t + 1:] = np.maximum(w[t + 1:], need)
                if q == 0 and t_empty is None:
                    t_empty = t
                if t_empty is not None and t >= t_empty + cool:
                    w[t + 1:] = base_w[t + 1:]; active = False; t_on = None; t_empty = None
        return w
    out["reactive"] = reactive(np.full(H, 3))
    fw = np.full(H, 3); s = int(round(onset + eps)); nf = int(math.ceil(amp_pred / MU))
    fw[max(0, s):max(0, s + dur + 20)] = nf
    out["forecast"] = fw
    out["forecast+reactive"] = reactive(fw)
    return out

def mc(sigma_t, sigma_a, trials=2000, false_alarm=0.0):
    res = {k: dict(peak=[], expo=[], ws=[]) for k in ("fixed", "reactive", "forecast", "forecast+reactive")}
    for _ in range(trials):
        onset = int(rng.uniform(150, 210)); dur = 60
        amp = rng.uniform(24000, 36000)
        burst = rng.random() >= false_alarm
        lam = np.full(H, BASE)
        if burst:
            lam[onset:onset + dur] = amp
        arr = rng.poisson(lam).astype(float)
        eps = rng.normal(0, sigma_t) if sigma_t > 0 else 0.0
        amp_pred = amp * (1 + rng.normal(0, sigma_a)) if sigma_a > 0 else amp
        for k, w in policies(arr, onset, dur, amp_pred, eps).items():
            pk, ex, ws = run_schedule(arr, w)
            res[k]["peak"].append(pk); res[k]["expo"].append(ex); res[k]["ws"].append(ws)
    summ = {}
    for k, v in res.items():
        summ[k] = dict(peak_mean=float(np.mean(v["peak"])), peak_p95=float(np.percentile(v["peak"], 95)),
                       expo_mean=float(np.mean(v["expo"])), ws_mean=float(np.mean(v["ws"])),
                       zero_backlog_frac=float(np.mean(np.array(v["peak"]) < 5000)))
    return summ
OUT["A_mc"] = {f"timing_sd={st}s, amplitude_sd={int(sa*100)}%": mc(st, sa) for st, sa in ((0, 0.0), (10, 0.10), (30, 0.15))}
print("A done")

# ===================================================================== B  checkpoint interval
def exact_overhead(I, C, f, R):
    """Expected wall time per unit useful work for segments of I work + C checkpoint, exponential failures,
    restart R after each failure (no failures during restart): E[T_seg] = (1/f + R)(e^{f(I+C)} - 1)."""
    return (1 / f + R) * (math.exp(f * (I + C)) - 1) / I - 1
def first_order(I, C, f, R=0.0):
    return C / I + f * I / 2
def young(C, f): return math.sqrt(2 * C / f)
def daly(C, f):
    M = 1 / f
    if C >= 2 * M: return M
    return math.sqrt(2 * C * M) * (1 + math.sqrt(C / (2 * M)) / 3 + (C / (2 * M)) / 9) - C
def mc_overhead(I, C, f, R, segments=200_000, seed=5):
    g = np.random.default_rng(seed); total = 0.0
    need = I + C; done = 0
    # vectorised renewal: each segment repeats until a failure-free attempt
    times = np.zeros(segments); remaining = np.ones(segments, bool)
    while remaining.any():
        n = remaining.sum(); x = g.exponential(1 / f, n)
        ok = x >= need
        add = np.where(ok, need, x + R)
        times[remaining] += add
        idx = np.nonzero(remaining)[0]; remaining[idx[ok]] = False
    return times.mean() / I - 1
B = []
for name, C, f, R in (("base: C=1 s, MTBF=1 h, restart 0 s", 1.0, 1 / 3600, 0.0),
                      ("base + 30 s restart", 1.0, 1 / 3600, 30.0),
                      ("stress: C=10 s, MTBF=10 min, restart 60 s", 10.0, 1 / 600, 60.0)):
    grid = np.arange(5, 1201, 1.0)
    ex = np.array([exact_overhead(I, C, f, R) for I in grid])
    i_opt = float(grid[ex.argmin()])
    rows = []
    for I in (10, 30, 60, 90, 120, 300):
        rows.append(dict(I=I, first_order_pct=100 * first_order(I, C, f), exact_pct=100 * exact_overhead(I, C, f, R),
                         mc_pct=100 * mc_overhead(I, C, f, R)))
    B.append(dict(scenario=name, C=C, mtbf_s=1 / f, R=R, young=young(C, f), daly=daly(C, f), exact_opt=i_opt,
                  exact_at_young_pct=100 * exact_overhead(young(C, f), C, f, R),
                  exact_at_opt_pct=100 * exact_overhead(i_opt, C, f, R),
                  first_order_at_young_pct=100 * first_order(young(C, f), C, f), rows=rows,
                  curve=dict(I=grid[::5].tolist(), exact=(100 * ex[::5]).tolist(),
                             first=[100 * first_order(I, C, f) for I in grid[::5]])))
OUT["B"] = B
print("B done")

# ===================================================================== C  watermarks
def watermark_experiment(minutes=30, rate=2000, keys=500, win=60.0):
    n = int(minutes * 60 * rate)
    et = np.sort(rng.uniform(0, minutes * 60, n))
    w = 1 / np.arange(1, keys + 1) ** 0.8; w /= w.sum()
    key = rng.choice(keys, n, p=w)
    straggler = rng.random(n) < 0.01
    delay = np.where(straggler, np.minimum(5.0 * (1 + rng.pareto(1.5, n)), 600.0), rng.lognormal(np.log(0.2), 1.0, n))
    at = et + delay
    o = np.argsort(at); et_s, at_s, key_s = et[o], at[o], key[o]
    maxet = np.maximum.accumulate(et_s)
    wid = np.floor(et_s / win).astype(int)
    nwin = int((minutes * 60 - 180) / win)   # exclude final windows whose watermark may not pass within the horizon
    # decision threshold: per key-window counts, threshold at 90th percentile of final counts
    final_counts = np.zeros((nwin, keys))
    m = wid < nwin
    np.add.at(final_counts, (wid[m], key_s[m]), 1)
    tau = np.percentile(final_counts[final_counts > 0], 90)
    rows = []
    for L in (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0):
        wm = maxet - L
        ends = (np.arange(nwin) + 1) * win
        fidx = np.searchsorted(wm, ends, side="left")
        fidx = np.minimum(fidx, len(at_s) - 1)
        fire_t = at_s[fidx]
        latency = fire_t - ends
        late = m & (np.arange(len(at_s)) >= fidx[np.minimum(wid, nwin - 1)])
        prov = np.zeros((nwin, keys))
        on_time = m & ~late
        np.add.at(prov, (wid[on_time], key_s[on_time]), 1)
        for A in (0.0, 60.0):
            accepted_late = late & (wm < (wid + 1) * win + A)
            dropped = late & ~accepted_late
            flips = ((prov < tau) & (final_counts >= tau)).sum() / max(1, (final_counts >= tau).sum())
            rows.append(dict(watermark_lag_s=L, allowed_lateness_s=A,
                             mean_result_latency_s=float(latency.mean()),
                             completeness_at_first_result_pct=float(100 * on_time.sum() / m.sum()),
                             late_pct=float(100 * late.sum() / m.sum()),
                             dropped_pct=float(100 * dropped.sum() / m.sum()),
                             incomplete_key_window_results_pct=float(100 * ((prov != final_counts) & (final_counts > 0)).sum() / (final_counts > 0).sum()),
                             decision_flips_pct=float(100 * flips)))
    delay_q = dict(p50=float(np.median(delay)), p99=float(np.percentile(delay, 99)), p999=float(np.percentile(delay, 99.9)))
    return rows, delay_q, float(tau)
rowsC, dq, tau = watermark_experiment()
OUT["C"] = dict(rows=rowsC, delay_quantiles_s=dq, decision_threshold=tau)
print("C done")
json.dump(OUT, open("results.json", "w"), indent=1)

# ------------------------------------------------------------------ print summaries
for r in OUT["A_grid"]: print("A grid", r)
for k, v in OUT["A_mc"].items():
    print("A mc", k); [print("   ", p, {kk: round(vv, 3) for kk, vv in s.items()}) for p, s in v.items()]
for b in OUT["B"]:
    print("B", b["scenario"], "young", round(b["young"], 1), "daly", round(b["daly"], 1), "exact", b["exact_opt"],
          "exact@young", round(b["exact_at_young_pct"], 3), "exact@opt", round(b["exact_at_opt_pct"], 3), "first@young", round(b["first_order_at_young_pct"], 3))
    for r in b["rows"]: print("    ", {k: round(v, 3) for k, v in r.items()})
print("C delays", dq, "tau", tau)
for r in rowsC: print("C", {k: round(v, 3) for k, v in r.items()})

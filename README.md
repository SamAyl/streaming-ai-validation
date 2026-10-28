# Real-Time Stream Processing for AI — robustness and validation experiments

Code and outputs for the stochastic experiments added to Section 5 of:

> S. Ayloo, *Real-Time Stream Processing Architectures for Scalable AI-Enabled Applications* (manuscript under submission, 2026).

| Experiment | Paper | Question |
|---|---|---|
| A | Table 5, Fig. 4a | How do forecast timing and amplitude errors change forecast-assisted provisioning? Fixed vs reactive vs forecast vs forecast + reactive, Poisson arrivals, 2,000 trials per error level. |
| B | Table 6, Fig. 4b | How accurate is the first-order checkpoint objective `J(I) = C/I + fI/2`? Exact renewal model under exponential failures with restart time, Young and Daly optima, Monte Carlo check. |
| C | Table 7, Fig. 4c | How does bounded-out-of-orderness watermark lag trade result latency against per-entity completeness, decision changes, and dropped late events? |

## Run

```bash
pip install -r requirements.txt
python stream_experiments.py   # ~1 min -> results.json
python stream_figure.py        # figures/fig4.png
```

All randomness is seeded; `results.json` is the exact output reported in the paper.
These are simulations with stated parameters, not measurements of deployed systems.

## License

MIT

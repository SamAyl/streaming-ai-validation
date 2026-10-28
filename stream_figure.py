"""Figure 4 for the streaming-AI manuscript additions, drawn from results.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("results.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.6, "axes.spines.top": False, "axes.spines.right": False})
DARK, MID, ACC, GREY = "#1f3b57", "#5b7a99", "#b5532e", "#9aa9b8"
fig, ax = plt.subplots(3, 1, figsize=(3.5, 8.4), dpi=300)

# (a) forecast robustness: mean backlog exposure by policy and error level
lvls = list(R["A_mc"].keys()); short = ["none", "10 s\n10%", "30 s\n15%"]
pols = [("reactive", "Reactive", GREY), ("forecast", "Forecast", ACC), ("forecast+reactive", "Forecast + reactive", DARK)]
x = np.arange(len(lvls)); w = 0.26
for j, (k, lab, c) in enumerate(pols):
    v = [R["A_mc"][l][k]["expo_mean"] / 1e6 for l in lvls]
    ax[0].bar(x + (j - 1) * w, v, w, color=c, label=lab)
ax[0].set_xticks(x, ["none", "10 s / 10%", "30 s / 15%"], fontsize=7); ax[0].set_xlabel("Forecast error sd (timing / amplitude)", fontsize=7.4)
ax[0].set_ylabel("Mean exposure (M event-s)"); ax[0].legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
ax[0].set_title("(a) Forecast error", fontsize=8.6, loc="left")

# (b) checkpoint: exact vs first-order
for b, c, lab in ((R["B"][1], MID, "C=1 s, MTBF 1 h, R=30 s"), (R["B"][2], ACC, "C=10 s, MTBF 10 min, R=60 s")):
    I = np.array(b["curve"]["I"]); m = I <= 600
    ax[1].plot(I[m], np.array(b["curve"]["exact"])[m], color=c, lw=1.5, label=lab + ", exact")
    ax[1].plot(I[m], np.array(b["curve"]["first"])[m], color=c, lw=1.1, ls="--")
    ax[1].axvline(b["exact_opt"], color=c, lw=0.6, ls=":")
ax[1].plot([], [], "k--", lw=1.1, label="first-order J(I)")
ax[1].set_ylim(0, 60); ax[1].set_xlabel("Checkpoint interval I (s)"); ax[1].set_ylabel("Lost-time fraction (%)")
ax[1].legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1); ax[1].set_title("(b) Checkpoint interval", fontsize=8.6, loc="left")

# (c) watermarks
rows = [r for r in R["C"]["rows"] if r["allowed_lateness_s"] == 0.0]
L = [r["watermark_lag_s"] for r in rows]
ax[2].plot(L, [r["incomplete_key_window_results_pct"] for r in rows], "o-", color=DARK, ms=3.5, lw=1.3, label="key-windows with incomplete first result")
ax[2].plot(L, [r["late_pct"] for r in rows], "s--", color=ACC, ms=3.2, lw=1.1, label="late events (dropped without allowed lateness)")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xticks(L, [str(v).rstrip("0").rstrip(".") for v in L], fontsize=7); ax[2].minorticks_off()
ax[2].set_xlabel("Watermark lag L (s)"); ax[2].set_ylabel("Share (%)")
ax[2].legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
ax[2].set_title("(c) Watermark lag", fontsize=8.6, loc="left")
fig.tight_layout(h_pad=1.2); fig.savefig("fig4.png"); print("fig4.png written")

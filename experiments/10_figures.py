"""Generate the paper figures from stored results and cached runs into paper/figures/."""
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from devsig.plots import BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID, COL1, COL2, style, seq_cmap, save  # noqa: E402
from devsig.signature import Signatures  # noqa: E402
from devsig.baselines import RegressionResidual  # noqa: E402
from devsig.threshold import NullModel  # noqa: E402
b7 = importlib.import_module("07_dynamic_signatures")

FIG = os.path.join(HERE, "..", os.environ["DEVSIG_RESULTS"], "figures") if "DEVSIG_RESULTS" in os.environ else os.path.join(HERE, "..", "paper", "figures")
RES = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"))
os.makedirs(FIG, exist_ok=True)
style()


STILL = os.path.join(HERE, "..", "results_still", "runs", "tier_a")  # the running example is flown without turbulence


def load_still(ac, variant, man, seeds):
    from devsig.run import load_run
    ps = [os.path.join(STILL, ac, f"{variant}_{man}_{s:04d}.parquet") for s in seeds]
    return [load_run(p) for p in ps if os.path.exists(p)]


def fig_doublet_example():
    base = load_still("c310", "baseline", "elev_doublet", range(20))
    cand = load_still("c310", "cmq_x0.5", "elev_doublet", range(100, 110))
    n = min(len(r) for r in base + cand)
    t = base[0]["t"].to_numpy()[:n] - base[0]["t"].to_numpy()[0]
    q = np.stack([np.degrees(r["velocities/q-rad_sec"].to_numpy()[:n]) for r in base])
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(COL1, 2.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.6], hspace=0.12))
    same = load_still("c310", "baseline", "elev_doublet", [100])[0]  # the correct run with the candidate's draw
    a0.plot(t, same["fcs/elevator-cmd-norm"].to_numpy()[:n], color=INK2, lw=1.2)
    a0.set_ylabel("elevator\ncmd"); a0.set_ylim(-0.35, 0.15); a0.set_yticks([-0.2, 0.0])
    a1.fill_between(t, q.min(axis=0), q.max(axis=0), color=GRID, lw=0, label="range of 20 correct runs")
    a1.plot(t, np.degrees(same["velocities/q-rad_sec"].to_numpy()[:n]), color=BLUE, label="correct run, same draw")
    a1.plot(t, np.degrees(cand[0]["velocities/q-rad_sec"].to_numpy()[:n]), color=ORANGE, label="new version, pitch damping halved")
    a1.set_xlabel("time (s)"); a1.set_ylabel("pitch rate (deg/s)"); a1.set_xlim(0, 20)
    a1.legend(loc="upper right", fontsize=7, handlelength=1.6)
    save(fig, os.path.join(FIG, "fig_doublet_example.pdf"))


def fig_score_null():
    """The decision on the running example (no turbulence): per-run residual scores of the 20 held-out correct runs
    and the 10 candidate runs, and the permutation distribution of median(candidate) - mean(held-out)."""
    base = load_still("c310", "baseline", "elev_doublet", range(40))
    train, held = base[:len(base) // 2], base[len(base) // 2:]  # the split of the detection pipeline
    cand = load_still("c310", "cmq_x0.5", "elev_doublet", range(100, 110))
    det = RegressionResidual(max_iter=60).fit(train)
    m = "velocities/q-rad_sec"
    H = pd.DataFrame([det.score(r) for r in held])[m].to_numpy(); C = pd.DataFrame([det.score(r) for r in cand])[m].to_numpy()
    mu = H.mean(); H, C = H - mu, C - mu
    obs = np.median(C) - H.mean()
    pooled = np.concatenate([C, H]); rng = np.random.default_rng(0)
    nc = len(C); perm = np.array([np.median(pooled[i[:nc]]) - pooled[i[nc:]].mean() for i in (rng.permutation(len(pooled)) for _ in range(5000))])
    pval = (1 + (perm >= obs - 1e-12).sum()) / 5001
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(COL2, 1.9), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
    j = np.random.default_rng(1)
    a0.scatter(H, j.uniform(-0.15, 0.15, len(H)) + 1, s=16, color=BLUE, edgecolor="white", lw=0.5, zorder=3)
    a0.scatter(C, j.uniform(-0.15, 0.15, len(C)), s=16, color=ORANGE, edgecolor="white", lw=0.5, zorder=3)
    a0.plot([np.median(C)] * 2, [-0.3, 0.3], color=ORANGE, lw=1.6); a0.plot([0, 0], [0.7, 1.3], color=BLUE, lw=1.6)
    a0.text(np.median(C), -0.42, "median", color=ORANGE, fontsize=7, ha="center", va="top")
    a0.text(0, 1.42, "mean", color=BLUE, fontsize=7, ha="center", va="bottom")
    a0.set_yticks([0, 1]); a0.set_yticklabels(["10 runs of the\nnew version", "20 held-out\ncorrect runs"]); a0.set_ylim(-0.8, 1.8); a0.grid(axis="y", visible=False)
    a0.set_xlabel("residual score, pitch rate")
    a1.hist(perm, bins=40, color=GRID, edgecolor="white", lw=0.3)
    a1.axvline(obs, color=ORANGE, lw=1.6)
    a1.text(obs, a1.get_ylim()[1] * 0.95, f"observed\n$p$ = {pval:.4f}" if pval >= 1e-3 else "observed\n$p$ < 0.001", color=ORANGE, fontsize=7, ha="right", va="top")
    a1.set_xlabel("median minus mean, 5,000 regroupings"); a1.set_yticks([]); a1.grid(axis="y", visible=False)
    for sp in ("left",):
        a1.spines[sp].set_visible(False)
    save(fig, os.path.join(FIG, "fig_score_null.pdf"))
    print("decision example: observed", round(float(obs), 4), "p", pval, "perm 95th", round(float(np.quantile(perm, 0.95)), 4))


def fig_effect_size():
    d = pd.read_csv(os.path.join(RES, "05_c310_graded.csv"))
    # rebuild per-metric effect sizes is expensive; use the max columns already there per fault as a 2-col heatmap
    d = d.set_index("fault")[["es_takeoff_max", "es_cruise_max"]]
    order = ["inertia_x1.15", "inertia_x1.3", "inertia_x1.6", "inertia_x2.0", "pitchnl_s0.3", "pitchnl_s0.0", "pitchnl_s-0.5",
             "clalpha_x0.8", "clalpha_x1.2", "cmq_x0.5", "cmalpha_x0.7", "cmalpha_x0.4", "lift_+0.05", "lift_+0.10", "lift_+0.20"]
    d = d.reindex([o for o in order if o in d.index])
    labels = {"inertia_x1.15": "inertia x1.15", "inertia_x1.3": "inertia x1.3", "inertia_x1.6": "inertia x1.6", "inertia_x2.0": "inertia x2.0",
              "pitchnl_s0.3": "Cm nonlinear, slope 0.3", "pitchnl_s0.0": "Cm nonlinear, slope 0", "pitchnl_s-0.5": "Cm nonlinear, slope -0.5",
              "clalpha_x0.8": "CL_alpha x0.8", "clalpha_x1.2": "CL_alpha x1.2", "cmq_x0.5": "Cm_q x0.5", "cmalpha_x0.7": "Cm_alpha x0.7",
              "cmalpha_x0.4": "Cm_alpha x0.4", "lift_+0.05": "CL_0 +0.05", "lift_+0.10": "CL_0 +0.10", "lift_+0.20": "CL_0 +0.20"}
    fig, ax = plt.subplots(figsize=(COL1, 3.2))
    v = np.log10(d.to_numpy().clip(0.05, 50))
    im = ax.imshow(v, cmap=seq_cmap(), aspect="auto", vmin=np.log10(0.3), vmax=np.log10(20))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["take-off\n(2-120 s)", "cruise\n(120-1800 s)"])
    ax.set_yticks(range(len(d))); ax.set_yticklabels([labels[i] for i in d.index]); ax.grid(False)
    for i in range(len(d)):
        for j in range(2):
            val = d.iloc[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="white" if val > 3 else INK)
    ax.axhline(-0.5, color="white"); 
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, ticks=[np.log10(0.3), 0, np.log10(3), 1])
    cb.ax.set_yticklabels(["0.3", "1", "3", "10"]); cb.set_label("effect size (fault shift / benign variation)")
    save(fig, os.path.join(FIG, "fig_effect_size_c310.pdf"))


def fig_rq2_auc():
    d = pd.read_csv(os.path.join(RES, "08_baselines_rq2.csv"))
    d = d[d.fault != "noop"]
    faults = ["cmalpha_x0.7", "cmq_x0.5", "clo_+0.1", "inertia_x1.6"]
    flabel = {"cmalpha_x0.7": "Cm_alpha x0.7", "cmq_x0.5": "Cm_q x0.5", "clo_+0.1": "CL_0 +0.1", "inertia_x1.6": "inertia x1.6"}
    methods = ["regression", "envelope", "sig_dynamic", "sig_static", "ks", "iforest"]
    mlabel = {"regression": "regression residual", "envelope": "envelope", "sig_dynamic": "dynamic signatures", "sig_static": "static signatures", "ks": "KS test", "iforest": "isolation forest"}
    g = d.groupby(["aircraft", "method", "fault"])["auc"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 3.6), sharey=True, sharex=True, gridspec_kw=dict(wspace=0.08, hspace=0.35))
    axes = axes.ravel()
    for ax, f in zip(axes, faults):
        for i, m in enumerate(methods):
            y = len(methods) - 1 - i
            for ac, col, mk in (("c310", BLUE, "o"), ("737", ORANGE, "s")):
                if (ac, m, f) in g.index:
                    ax.scatter(g[(ac, m, f)], y, s=22, color=col, marker=mk, edgecolor="white", lw=0.5, zorder=3, label=ac if (i == 0 and f == faults[0]) else None)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls="-"); ax.set_xlim(0.3, 1.0); ax.set_title(flabel[f], fontsize=8)
        ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", "0.75", "1"])
        ax.set_yticks(range(len(methods))); ax.set_yticklabels([mlabel[m] for m in reversed(methods)]); ax.grid(axis="y", visible=False)
    for ax in axes[2:]:
        ax.set_xlabel("AUC")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.06), handletextpad=0.2, columnspacing=1.5)
    save(fig, os.path.join(FIG, "fig_rq2_auc.pdf"))


def fig_static_vs_dynamic():
    """Baseline rule-violation rate, static vs dynamic signatures, per manoeuvre: median over the 25 aircraft with quartile bars."""
    d = pd.read_csv(os.path.join(HERE, "..", "results_still", "07_dynamic_signatures.csv"))  # no turbulence, as the text
    d = d[d.config.isin(["static", "lags+since"])]
    mans = list(b7.BATTERY.keys())
    mlabel = {"cruise": "cruise", "throttle_step": "throttle step", "elev_doublet": "elevator doublet", "elev_step": "elevator step",
              "aileron_step": "aileron step", "rudder_doublet": "rudder doublet", "bank_hold": "sustained bank", "flap_extend": "flap extension", "engine_cut": "engine cut",
              "climbing_turn": "climbing turn", "elev_doublet_distractor": "doublet + distractors", "engine_restart": "engine restart", "engine_stop": "engine stop"}
    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    x = np.arange(len(mans))
    for cfg, col, dx, lab in (("static", BLUE, -0.12, "static signatures"), ("lags+since", ORANGE, 0.12, "dynamic signatures")):
        g = d[d.config == cfg].groupby("manoeuvre")["base_vsd"]
        med = g.median().reindex(mans); q1 = g.quantile(0.25).reindex(mans); q3 = g.quantile(0.75).reindex(mans)
        ax.vlines(x + dx, q1, q3, color=col, lw=1.4, alpha=0.5)
        ax.scatter(x + dx, med, color=col, s=24, zorder=3, edgecolor="white", lw=0.5, label=lab)
    ax.axhline(0.8, color=MUTED, lw=0.8); ax.text(len(mans) - 0.6, 0.81, "chance, five levels", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([mlabel[m] for m in mans], rotation=30, ha="right")
    ax.set_ylim(0, 0.9); ax.grid(axis="x", visible=False); ax.set_ylabel("rule violations on\nheld-out correct runs")
    leg = ax.legend(loc="lower left", fontsize=7, title="dot: median; bar: quartiles over the aircraft", title_fontsize=6.5)
    leg._legend_box.align = "left"
    save(fig, os.path.join(FIG, "fig_static_vs_dynamic.pdf"))


def fig_injected_vs_real():
    """Observability in both conditions. Left: share of outputs the detector flags by effect-size bin (residual and
    envelope; no turbulence solid, turbulence dashed). Right: per fault, median effect size against detector recall,
    no turbulence (filled) joined to turbulence (open); injected faults circles, real faults squares."""
    import json
    conds = {"no turbulence": os.path.join(RES, "..", "results_still"), "turbulence": RES}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.7), gridspec_kw=dict(wspace=0.38))
    bins = ["<1", "1-2", "2-4", "4-8", ">8"]
    for cond, ls in (("no turbulence", "-"), ("turbulence", "--")):
        a = json.load(open(os.path.join(conds[cond], "19_analysis.json")))
        for m, col, lab in (("regression", BLUE, "residual"), ("envelope", ORANGE, "envelope")):
            y = [a["flag_rate_by_effect_size"][m].get(b, np.nan) for b in bins]
            ax1.plot(range(len(bins)), y, ls=ls, marker="o", ms=3.5, color=col, lw=1.4, label=f"{lab}, {cond}")
    ax1.set_xticks(range(len(bins))); ax1.set_xticklabels(bins); ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("effect size of the output"); ax1.set_ylabel("share of outputs flagged")
    ax1.legend(fontsize=6, loc="lower right", handlelength=2.2)
    lab_i = {"clo_+0.1": "lift offset", "cmalpha_x0.7": "pitch stiffness", "cmq_x0.5": "pitch damping", "inertia_x1.6": "inertia"}
    lab_r = {"B_elevator_drag_8819410c": "elevator drag"}
    pts = {}
    for cond in conds:
        d = pd.read_csv(os.path.join(conds[cond], "08_baselines_rq2.csv")); d = d[(d.method == "regression") & (d.fault != "noop")]
        import glob
        f06 = sorted(glob.glob(os.path.join(conds[cond], "06_tier_a_battery_*.csv")), key=len)[-1]
        a6 = pd.read_csv(f06); a6 = a6[a6.fault != "noop"]
        for f in lab_i:
            pts[(f, cond)] = (float(a6[a6.fault == f].es_max.median()), float(d[d.fault == f].recall.mean()), "o")
        b = pd.read_csv(os.path.join(conds[cond], "13_family_b.csv"))
        b = b[~b.manoeuvre.isin(["engine_restart", "engine_stop"])]  # the manoeuvres of the upper rows of the real-fault table
        for v, g in b.groupby("variant"):
            pts[(v, cond)] = (float(g.es_max.median()), float(g.phys_recall.mean()), "s")
    ax2.axvline(1.0, color=MUTED, lw=0.8)
    keys = sorted({k for k, _ in pts})
    for k in keys:
        (x1, y1, mk), (x2, y2, _) = pts[(k, "no turbulence")], pts[(k, "turbulence")]
        col = BLUE if mk == "o" else ORANGE
        ax2.plot([x1, x2], [y1, y2], color=col, lw=0.8, alpha=0.6, zorder=2)
        ax2.scatter([x1], [y1], s=30, marker=mk, color=col, edgecolor="white", lw=0.5, zorder=3)
        ax2.scatter([x2], [y2], s=30, marker=mk, facecolor="white", edgecolor=col, lw=1.0, zorder=3)
        lab = lab_i.get(k) or lab_r.get(k)
        if lab:
            ax2.annotate(lab, (x1, y1), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK2)
    ax2.set_xscale("symlog", linthresh=1.0, linscale=0.6); ax2.set_xlim(-0.05, 12); ax2.set_ylim(-0.03, 0.95)
    ax2.set_xticks([0, 0.5, 1, 2, 4, 8]); ax2.set_xticklabels(["0", "0.5", "1", "2", "4", "8"])
    ax2.set_xlabel("effect size, median over pairs"); ax2.set_ylabel("detector recall")
    ax2.scatter([], [], marker="o", color=BLUE, label="injected"); ax2.scatter([], [], marker="s", color=ORANGE, label="real")
    ax2.scatter([], [], marker="o", facecolor="white", edgecolor=INK2, label="turbulence (open)"); ax2.scatter([], [], marker="o", color=INK2, label="no turbulence (filled)")
    ax2.legend(fontsize=6, loc="upper left", ncol=2, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.4)
    save(fig, os.path.join(FIG, "fig_injected_vs_real.pdf"))


if __name__ == "__main__":
    for f in (fig_doublet_example, fig_score_null, fig_effect_size, fig_rq2_auc, fig_static_vs_dynamic, fig_injected_vs_real):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:  # noqa: BLE001
            print("FAILED", f.__name__, repr(e))

"""Turn the 25-aircraft results (08, 09) into paper figures and LaTeX tables."""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from devsig.plots import BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID, COL1, COL2, style, save  # noqa: E402

RES = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"))
FIG = os.path.join(HERE, "..", os.environ["DEVSIG_RESULTS"], "figures") if "DEVSIG_RESULTS" in os.environ else os.path.join(HERE, "..", "paper", "figures")
TAB = os.path.join(HERE, "..", os.environ["DEVSIG_RESULTS"], "tables") if "DEVSIG_RESULTS" in os.environ else os.path.join(HERE, "..", "paper", "tables")
os.makedirs(TAB, exist_ok=True)
style()
CLS = {k: v["cls"] for k, v in json.load(open(os.path.join(HERE, "..", "config", "tier_a_aircraft.json")))["tier_a"].items()}
CLS_LABEL = {"airliner": "airliner", "bizjet": "business jet", "fighter": "fighter", "trainer": "trainer", "turboprop": "turboprop",
             "light": "light piston", "cub": "light piston", "vintage": "vintage"}
METHODS = ["regression", "envelope", "sig_dynamic", "sig_static", "ks", "iforest"]
MLABEL = {"regression": "regression residual", "envelope": "envelope", "sig_dynamic": "dynamic signatures", "sig_static": "static signatures", "ks": "KS test", "iforest": "isolation forest"}
FAULTS = ["cmalpha_x0.7", "cmq_x0.5", "clo_+0.1", "inertia_x1.6"]
FLABEL = {"cmalpha_x0.7": "pitch stiffness x0.7", "cmq_x0.5": "pitch damping x0.5", "clo_+0.1": "lift offset +0.1", "inertia_x1.6": "inertia x1.6"}


def load08():
    d = pd.read_csv(os.path.join(RES, "08_baselines_rq2.csv"))
    d["cls"] = d.aircraft.map(CLS).map(CLS_LABEL)
    return d


def fig_rq2_all(d):
    """One dot per aircraft (mean over manoeuvres), rows = methods, facets = faults."""
    g = d[d.fault != "noop"].groupby(["fault", "method", "aircraft"])["auc"].mean().reset_index()
    faults = [f for f in FAULTS if g[g.fault == f]["auc"].notna().any()]  # AUC is undefined when the oracle covers every output
    fig, axes = plt.subplots(1, len(faults), figsize=(COL2, 2.4), sharey=True, sharex=True, gridspec_kw=dict(wspace=0.22))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, f in zip(axes.ravel(), faults):
        sub = g[g.fault == f]
        for i, m in enumerate(METHODS):
            y = len(METHODS) - 1 - i
            v = sub[sub.method == m]["auc"].to_numpy()
            if len(v) == 0:
                continue
            ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=9, color=BLUE, alpha=0.55, edgecolor="none", zorder=2)
            med = np.nanmedian(v)
            ax.plot([med, med], [y - 0.32, y + 0.32], color=ORANGE, lw=2, zorder=3)
        ax.axvline(0.5, color=MUTED, lw=0.8); ax.set_xlim(0.2, 1.0); ax.set_xticks([0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["", "0.5", "", "1"], fontsize=7)
        ax.set_title(f"{FLABEL[f]}\n{sub.aircraft.nunique()} aircraft", fontsize=7.5)
        ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([MLABEL[m] for m in reversed(METHODS)]); ax.grid(axis="y", visible=False)
    axes.ravel()[len(faults) // 2].set_xlabel("AUC, mean over the manoeuvres", fontsize=7.5)
    save(fig, os.path.join(FIG, "fig_rq2_auc_all.pdf"))


def fig_rq6_class(d):
    """Detector recall by aircraft class and fault."""
    sub = d[(d.method == "regression") & (d.fault != "noop")]
    g = sub.groupby(["cls", "fault"])["recall"].mean().unstack("fault").reindex(columns=FAULTS)
    n = sub.groupby("cls")["aircraft"].nunique()
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    im = ax.imshow(g.to_numpy(), cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f7fb", "#2a78d6"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FAULTS))); ax.set_xticklabels([FLABEL[f] for f in FAULTS], rotation=30, ha="right")
    ax.set_yticks(range(len(g))); ax.set_yticklabels([f"{c} (n={n[c]})" for c in g.index]); ax.grid(False)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("detector recall")
    save(fig, os.path.join(FIG, "fig_rq6_class.pdf"))


def tab_rq2(d):
    """LaTeX table: per method, mean recall / precision / AUC over aircraft and manoeuvres, per fault; plus noop false alarms."""
    sub = d[d.fault != "noop"]
    rows = []
    for m in METHODS:
        r = {"method": MLABEL[m]}
        for f in FAULTS:
            x = sub[(sub.method == m) & (sub.fault == f)]
            r[f] = f"{x.recall.mean():.2f}/{x.precision.mean():.2f}/{x.auc.mean():.2f}" if len(x) else "--"
        fa = d[(d.method == m) & (d.fault == "noop")]["false_alarm"].mean()
        r["fa"] = f"{100 * fa:.1f}\\%" if not np.isnan(fa) else "--"
        rows.append(r)
    gs = sub[sub.gated_recall.notna()] if "gated_recall" in sub else None
    lines = ["\\begin{tabular}{l" + "c" * len(FAULTS) + "cc}", "\\toprule", "method & " + " & ".join(FLABEL[f] for f in FAULTS) + " & no-op & gated \\\\",
             " & " + " & ".join(["R/P/AUC"] * len(FAULTS)) + " & false alarms & R/AUC \\\\", "\\midrule"]
    for r, m in zip(rows, METHODS):
        g = f"{gs[gs.method == m].gated_recall.mean():.2f}/{gs[gs.method == m].gated_auc.mean():.2f}" if gs is not None and len(gs) else "--"
        lines.append(r["method"] + " & " + " & ".join(r[f] for f in FAULTS) + " & " + r["fa"] + " & " + g + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "tab_rq2.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return rows


def tab_rq2_gated(d):
    """LaTeX table: per method, mean gated recall / precision / AUC per fault (outputs above the effect-size gate only)."""
    sub = d[(d.fault != "noop") & d.gated_recall.notna()]
    lines = ["\\begin{tabular}{l" + "c" * len(FAULTS) + "c}", "\\toprule", "method & " + " & ".join(FLABEL[f] for f in FAULTS) + " & mean \\\\",
             " & " + " & ".join(["R/P/AUC"] * len(FAULTS)) + " & AUC \\\\", "\\midrule"]
    for m in METHODS:
        cells = []
        for f in FAULTS:
            x = sub[(sub.method == m) & (sub.fault == f)]
            cells.append(f"{x.gated_recall.mean():.2f}/{x.gated_precision.mean():.2f}/{x.gated_auc.mean():.2f}" if len(x) else "--")
        lines.append(MLABEL[m] + " & " + " & ".join(cells) + f" & {sub[sub.method == m].gated_auc.mean():.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "tab_rq2_gated.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


MAN_LABEL = {"aileron_step": "aileron step", "bank_hold": "sustained bank", "elev_doublet": "elevator doublet", "elev_step": "elevator step",
             "engine_cut": "engine cut", "flap_extend": "flap extension", "rudder_doublet": "rudder doublet", "throttle_step": "throttle step", "cruise": "cruise"}


def tab_rq2b(e):
    """LaTeX table: per manoeuvre, pooled agreement and correctness of the dynamic signatures (shares of all flags)."""
    x = e[e.explainer == "dynamic"]
    lines = ["\\begin{tabular}{lrcc}", "\\toprule", "manoeuvre & flags & agreement & correctness \\\\", "\\midrule"]
    for man, g in x.groupby("manoeuvre"):
        if man == "cruise":
            continue
        a = (g.agreement * g.n_flagged).sum() / g.n_flagged.sum(); c = (g.correctness * g.n_explained).sum() / g.n_explained.sum()
        lines.append(f"{MAN_LABEL.get(man, man)} & {int(g.n_flagged.sum())} & {a:.2f} & {c:.2f} \\\\")
    y = x[x.manoeuvre != "cruise"]
    lines += ["\\midrule", f"all eight & {int(y.n_flagged.sum())} & {(y.agreement * y.n_flagged).sum() / y.n_flagged.sum():.2f} & {(y.correctness * y.n_explained).sum() / y.n_explained.sum():.2f} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "tab_rq2b.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def numbers(d):
    """Headline numbers for the text, as a JSON file the paper's todo marks are filled from."""
    sub = d[d.fault != "noop"]
    det = sub[sub.method == "regression"]
    out = dict(n_aircraft=int(d.aircraft.nunique()), n_pairs=int(sub.groupby(["aircraft", "manoeuvre"]).ngroups),
               detector_recall=round(float(det.recall.mean()), 2), detector_precision=round(float(det.precision.mean()), 2), detector_auc=round(float(det.auc.mean()), 2),
               detector_recall_by_fault={f: round(float(det[det.fault == f].recall.mean()), 2) for f in FAULTS},
               noop_false_alarm={m: round(float(d[(d.method == m) & (d.fault == "noop")].false_alarm.mean()), 3) for m in METHODS},
               auc_by_method={m: round(float(sub[sub.method == m].auc.mean()), 2) for m in METHODS})
    if "gated_recall" in sub:
        gs = sub[sub.gated_recall.notna()]
        out.update(gated_detector_recall=round(float(gs[gs.method == "regression"].gated_recall.mean()), 2),
                   gated_detector_precision=round(float(gs[gs.method == "regression"].gated_precision.mean()), 2),
                   gated_detector_auc=round(float(gs[gs.method == "regression"].gated_auc.mean()), 2),
                   gated_recall_by_fault={f: round(float(gs[(gs.method == "regression") & (gs.fault == f)].gated_recall.mean()), 2) for f in FAULTS},
                   gated_auc_by_method={m: round(float(gs[gs.method == m].gated_auc.mean()), 2) for m in METHODS},
                   gated_recall_by_method={m: round(float(gs[gs.method == m].gated_recall.mean()), 2) for m in METHODS},
                   frac_pairs_with_gated_outputs=round(float((sub.groupby(["aircraft", "manoeuvre", "fault"]).n_gated.first() > 0).mean()), 2))
    # precision is undefined when a test case flags nothing; report that share and the precision among cases that flag
    out["detector_share_no_flag"] = round(float((det.n_flagged == 0).mean()), 2)
    out["detector_precision_among_flagging"] = round(float(det[det.n_flagged > 0].precision.mean()), 2)
    out["share_no_flag_by_method"] = {m: round(float((sub[sub.method == m].n_flagged == 0).mean()), 2) for m in METHODS}
    p09 = os.path.join(RES, "09_explain_rq2b.csv")
    if os.path.exists(p09):
        e = pd.read_csv(p09)
        for k in ("dynamic", "static"):
            x = e[e.explainer == k]
            out[f"agreement_{k}"] = round(float(x.agreement.mean()), 2)          # mean over test cases
            out[f"correctness_{k}"] = round(float(x.correctness.mean()), 2)
            out[f"agreement_pooled_{k}"] = round(float((x.agreement * x.n_flagged).sum() / x.n_flagged.sum()), 2)  # share of all flags
            out[f"correctness_pooled_{k}"] = round(float((x.correctness * x.n_explained).sum() / x.n_explained.sum()), 2)
            out[f"trim_shift_share_{k}"] = round(float(x.n_with_trim_shift.sum() / x.n_explained.sum()), 2)
        tab_rq2b(e)
    with open(os.path.join(RES, "headline_numbers.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    d = load08()
    fig_rq2_all(d); fig_rq6_class(d); tab_rq2(d); tab_rq2_gated(d)
    print(json.dumps(numbers(d), indent=1))



def fig_tier_b_units():
    """Tier B: detector recall by evaluation unit (whole run, 30 s and 120 s windows, per phase) per fault variant."""
    o = pd.read_csv(os.path.join(RES, "16_tier_b.csv")); d = o[o.variant != "noop"]
    modes = ["det_whole", "det_win30", "det_win120", "det_phase_any"]
    mlab = {"det_whole": "whole run", "det_win30": "30 s windows", "det_win120": "120 s windows", "det_phase_any": "per phase"}
    variants = ["clo_+0.1", "cmalpha_x0.7", "cmq_x0.5", "inertia_x1.6", "B_elevator_drag_8819410c"]
    vlab = {"clo_+0.1": "lift offset +0.1", "cmalpha_x0.7": "pitch stiffness x0.7", "cmq_x0.5": "pitch damping x0.5", "inertia_x1.6": "inertia x1.6", "B_elevator_drag_8819410c": "elevator drag (real, 2020)"}
    p = d[d["mode"].isin(modes)].pivot_table(index="variant", columns="mode", values="recall", aggfunc="mean").reindex(index=variants, columns=modes)
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    x = np.arange(len(variants)); w = 0.2
    cols = [MUTED, "#9ec5f4", "#5598e7", BLUE]
    for i, m in enumerate(modes):
        ax.bar(x + (i - 1.5) * w, p[m].to_numpy(), width=w * 0.92, color=cols[i], label=mlab[m], edgecolor="white", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels([vlab[v] for v in variants], rotation=25, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("detector recall")
    ax.grid(axis="x", visible=False); ax.legend(loc="upper right", ncol=2, fontsize=6.5)
    save(fig, os.path.join(FIG, "fig_tier_b_units.pdf"))


if __name__ == "__main__" and "--tierb" in sys.argv:
    fig_tier_b_units(); print("ok fig_tier_b_units")


def tab_rq4():
    """LaTeX table for RQ4 from 16_tier_b.csv: detector and dynamic signature, whole-run vs windows vs phases."""
    d = pd.read_csv(os.path.join(RES, "16_tier_b.csv"))
    noop = d[d.variant == "noop"].groupby("mode")["false_alarm"].mean()
    d = d[d.variant != "noop"]
    rows = [("det_whole", "detector, whole run"), ("det_win30", "detector, 30 s windows"), ("det_win120", "detector, 120 s windows"), ("det_phase_any", "detector, any phase"),
            ("sig_dynamic_whole", "dynamic signatures, whole run"), ("sig_dynamic_win30", "dynamic signatures, 30 s windows"), ("sig_dynamic_phase_any", "dynamic signatures, any phase"),
            ("envelope_whole", "envelope, whole run")]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule", "scoring & recall & precision & gated recall & no-op false alarms \\\\", "\\midrule"]
    for m, lab in rows:
        x = d[d["mode"] == m]
        if not len(x):
            continue
        lines.append(f"{lab} & {x.recall.mean():.2f} & {x.precision.mean():.2f} & {x.gated_recall.mean():.2f} & {100 * noop.get(m, float('nan')):.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "tab_rq4.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # per aircraft, detector whole vs phase
    w = d[d["mode"].isin(["det_whole", "det_phase_any"])].pivot_table(index="aircraft", columns="mode", values="recall", aggfunc="mean").round(2)
    return w


if __name__ == "__main__" and "--rq4" in sys.argv:
    print(tab_rq4().to_string()); print(open(os.path.join(TAB, "tab_rq4.tex")).read())

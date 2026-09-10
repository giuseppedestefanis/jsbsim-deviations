"""Side-by-side tables for the two benign-variation conditions: no turbulence (results_still/) and moderate
turbulence (results/). Reads 08_baselines_rq2.csv, 09_explain_rq2b.csv, 13_family_b.csv, 16_tier_b.csv and
19_analysis.json from both, and writes paper/tables/tab_rq1_conditions.tex (six detectors x two conditions),
tab_rq3_attribution.tex (attribution measures x two conditions) and results/conditions.json (headline numbers)."""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COND = {"still": os.path.join(HERE, "..", "results_still"), "turb": os.path.join(HERE, "..", "results")}
TAB = os.path.join(HERE, "..", "paper", "tables")
METHODS = ["regression", "envelope", "sig_dynamic", "sig_static", "ks", "iforest"]
MLABEL = {"regression": "regression residual", "envelope": "envelope", "sig_dynamic": "dynamic signatures",
          "sig_static": "static signatures", "ks": "KS test", "iforest": "isolation forest"}
BATTERY9 = ["cruise", "throttle_step", "elev_doublet", "elev_step", "aileron_step", "rudder_doublet", "bank_hold", "flap_extend", "engine_cut"]
FAULTS = ["clo_+0.1", "cmalpha_x0.7", "cmq_x0.5", "inertia_x1.6"]


def f2(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def rq1(cond):
    p = os.path.join(COND[cond], "08_baselines_rq2.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p); d = d[d.manoeuvre.isin(BATTERY9)]
    sub = d[d.fault != "noop"]
    out = {}
    for m in METHODS:
        x = sub[sub.method == m]; fa = d[(d.method == m) & (d.fault == "noop")].false_alarm.mean()
        # precision against the primary oracle without the inertia instance, whose oracle covers every output
        out[m] = dict(recall=x.recall.mean(), precision=x[x.fault != "inertia_x1.6"].precision.mean(), precision_all=x.precision.mean(), auc=x.auc.mean(), gated_recall=x.gated_recall.mean(),
                      gated_precision=x.gated_precision.mean(), gated_auc=x.gated_auc.mean(), fa=fa, no_flag=(x.n_flagged == 0).mean())
    det = sub[sub.method == "regression"]
    out["_by_fault"] = {f: dict(recall=det[det.fault == f].recall.mean(), gated_recall=det[det.fault == f].gated_recall.mean(),
                                pairs_gated=(det[det.fault == f].n_gated > 0).mean()) for f in FAULTS}
    out["_n_pairs"] = int(sub.groupby(["aircraft", "manoeuvre"]).ngroups)
    return out


def rq3(cond):
    p = os.path.join(COND[cond], "09_explain_rq2b.csv")
    if not os.path.exists(p):
        return None
    e = pd.read_csv(p); out = {}
    for k in ("dynamic", "static"):
        x = e[e.explainer == k]; w = x.n_explained.clip(lower=0)
        pooled = lambda col: float((x[col] * w).sum() / w.sum()) if w.sum() else float("nan")
        out[k] = dict(agreement=float((x.agreement * x.n_flagged).sum() / x.n_flagged.sum()), agreement_null_median=float((x.agreement_null_median * x.n_flagged).sum() / x.n_flagged.sum()),
                      correctness=pooled("correctness"),
                      # baselines are per test case, so they are pooled over every detector flag (same denominator for both explainers)
                      largest_movement=float((x.correctness_largest_movement * x.n_flagged).sum() / x.n_flagged.sum()), only_changing=float((x.correctness_only_changing * x.n_flagged).sum() / x.n_flagged.sum()),
                      residual_importance=pooled("correctness_residual_importance") if k == "dynamic" else float("nan"),
                      stability=pooled("stability") if k == "dynamic" else float("nan"), trim_shift=float(x.n_with_trim_shift.sum() / w.sum()))
        for grp, mans in (("single_input", BATTERY9), ("multi_input", ["climbing_turn"]), ("distractor", ["elev_doublet_distractor"]), ("engine_restart", ["engine_restart"])):
            y = x[x.manoeuvre.isin(mans)]; wy = y.n_explained
            out[k][f"correctness_{grp}"] = float((y.correctness * wy).sum() / wy.sum()) if wy.sum() else float("nan")
            out[k][f"largest_movement_{grp}"] = float((y.correctness_largest_movement * y.n_flagged).sum() / y.n_flagged.sum()) if y.n_flagged.sum() else float("nan")
    return out


def main():
    res = {c: dict(rq1=rq1(c), rq3=rq3(c)) for c in COND}
    os.makedirs(TAB, exist_ok=True)
    # RQ1 table: two stacked panels, one per condition; rows = methods, then matched replay
    mr = {}
    for c in ("still", "turb"):
        pj = os.path.join(COND[c], "19_analysis.json")
        mr[c] = json.load(open(pj)).get("matched_replay") if os.path.exists(pj) else None
    lines = ["\\begin{tabular}{lcccccccc}", "\\toprule",
             "& \\multicolumn{3}{c}{primary oracle} & & \\multicolumn{3}{c}{gated oracle} & \\\\", "\\cmidrule(lr){2-4} \\cmidrule(lr){6-8}",
             "method & R & P & AUC & none & R & P & AUC & FA \\\\"]
    for c, title in (("still", "no turbulence"), ("turb", "moderate turbulence")):
        lines += ["\\midrule", "\\multicolumn{9}{l}{\\emph{" + title + "}} \\\\"]
        r = res[c]["rq1"]
        for m in METHODS:
            v = r[m] if r is not None else None
            cells = [MLABEL[m]] + ([f2(v["recall"]), f2(v["precision"]), f2(v["auc"]), f"{100 * v['no_flag']:.0f}\\%", f2(v["gated_recall"]), f2(v["gated_precision"]), f2(v["gated_auc"]), f"{100 * v['fa']:.1f}\\%"] if v else ["--"] * 8)
            lines.append(" & ".join(cells) + " \\\\")
        if mr[c]:
            lines.append("\\cmidrule(lr){1-9}")
            for key, label in (("any_difference", "matched replay, any difference"), ("beyond_benign_spread", "matched replay, beyond benign variation")):
                m = mr[c][key]; auc = mr[c]["ranking_matched"]["auc"]
                lines.append(" & ".join([label, f2(m["recall"]), f2(m["precision"]), f2(auc), "--", "--", "--", "--", f"{100 * m['noop_false_alarm']:.1f}\\%"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "tab_rq1_conditions.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # RQ3 table: attribution measures, dynamic signatures, both conditions (+ static correctness)
    rows = [("agreement with the detector (calibrated)", "agreement"),             ("correctness, all manoeuvres", "correctness"), ("correctness, single-input manoeuvres", "correctness_single_input"),
            ("correctness, climbing turn (multi-input)", "correctness_multi_input"), ("correctness, doublet with distractors", "correctness_distractor"),
            ("baseline: largest-movement command", "largest_movement"), ("baseline: largest movement, climbing turn", "largest_movement_multi_input"),
            ("baseline: largest movement, distractors", "largest_movement_distractor"),             ("residual permutation importance", "residual_importance"), ("stability across the ten runs", "stability"), ("flags with a trim shift", "trim_shift")]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule", "& \\multicolumn{2}{c}{no turbulence} & \\multicolumn{2}{c}{moderate turbulence} \\\\",
             "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}", "measure & dynamic & static & dynamic & static \\\\", "\\midrule"]
    for label, key in rows:
        cells = [label]
        for c in ("still", "turb"):
            r = res[c]["rq3"]
            if r is None:
                cells += ["--", "--"]
            elif key.startswith(("largest_movement", "only_changing", "residual_importance", "stability")):
                cells += ["\\multicolumn{2}{c}{" + f2(r["dynamic"].get(key)) + "}"]  # independent of the explainer: one value per condition
            else:
                cells += [f2(r["dynamic"].get(key)), f2(r["static"].get(key))]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "tab_rq3_attribution.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # bootstrap intervals, both conditions
    ci = {}
    for c in COND:
        pth = os.path.join(COND[c], "19_analysis.json")
        ci[c] = json.load(open(pth))["ci"] if os.path.exists(pth) else {}
    fmt = lambda c, k: (f"{ci[c][k][0]:.2f} [{ci[c][k][1]:.2f}, {ci[c][k][2]:.2f}]" if k in ci[c] else "--")
    rows = [("residual: recall", "regression:recall"), ("residual: AUC", "regression:auc"),
            ("residual: gated recall", "regression:gated_recall"), ("residual: gated AUC", "regression:gated_auc"),
            ("residual $-$ envelope: recall", "residual_minus_envelope:recall"), ("residual $-$ envelope: AUC", "residual_minus_envelope:auc"),
            ("residual $-$ envelope: gated recall", "residual_minus_envelope:gated_recall"), ("residual $-$ envelope: gated AUC", "residual_minus_envelope:gated_auc"),
            ("dynamic $-$ static rules: agreement", "dynamic_minus_static:agreement"), ("dynamic $-$ static rules: correctness", "dynamic_minus_static:correctness"),
            ("phase union $-$ whole run: recall", "det_phase_any_minus_det_whole:recall"), ("phase maximum $-$ whole run: recall", "det_phase_max_minus_det_whole:recall")]
    lines = ["\\begin{tabular}{lcc}", "\\toprule", "quantity & no turbulence & moderate turbulence \\\\", "\\midrule"]
    for label, k in rows:
        lines.append(f"{label} & {fmt('still', k)} & {fmt('turb', k)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "tab_ci_conditions.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # real faults (13) x conditions
    lab5 = {"B_elevator_drag_8819410c": "elevator drag (8819410c)", "B_ground_effect_9c058118": "ground-effect kink (9c058118)",
            "B_pqrdot_5ad2694c": "angular acceleration (5ad2694c)", "B_piston_power_bcd3f980": "piston negative power (bcd3f980)",
            "B_turbine_windmill_2db8e408": "turbine windmilling (2db8e408)"}
    b = {}
    COVERAGE = {"engine_restart": ("B_turbine_windmill_2db8e408", "turbine windmilling, engine restart"), "engine_stop": ("B_piston_power_bcd3f980", "piston negative power, engine stop")}
    agg = dict(n_aircraft=("aircraft", "nunique"), n_pairs=("manoeuvre", "size"), es=("es_max", "median"), gated=("n_gated", lambda x: float((x > 0).mean())),
               r=("phys_recall", "mean"), gr=("gated_recall", "mean"), auc=("phys_auc", "mean"), gauc=("gated_auc", "mean"), corr=("correct_dynamic", "mean"))
    for c in COND:
        pth = os.path.join(COND[c], "13_family_b.csv")
        if os.path.exists(pth):
            d = pd.read_csv(pth)
            bat = d[~d.manoeuvre.isin(COVERAGE)].groupby("variant").agg(**agg)  # the battery: nine QTG manoeuvres plus the two added
            cov = pd.concat([d[(d.manoeuvre == m) & (d.variant == v)].assign(variant=f"{v}:{m}").groupby("variant").agg(**agg) for m, (v, _) in COVERAGE.items()])
            p20 = os.path.join(COND[c], "20_coverage_low_pass.csv")
            if os.path.exists(p20):
                lp = pd.read_csv(p20); lp = lp[(lp.method == "regression") & (lp.variant != "noop")]
                if len(lp):
                    cov.loc["B_ground_effect_9c058118:low_pass"] = dict(n_aircraft=1, n_pairs=1, es=float(lp.es_max.iloc[0]), gated=float(lp.n_gated.iloc[0] > 0),
                                                                       r=float(lp.recall.iloc[0]), gr=float(lp.gated_recall.iloc[0]) if "gated_recall" in lp and pd.notna(lp.gated_recall.iloc[0]) else float("nan"),
                                                                       auc=float(lp.auc.iloc[0]), gauc=float("nan"), corr=float("nan"))
            b[c] = pd.concat([bat, cov])
    lab5.update({f"{v}:{m}": lab for m, (v, lab) in COVERAGE.items()}); lab5["B_ground_effect_9c058118:low_pass"] = "ground-effect kink, low pass"
    if b:
        lines = ["\\begin{tabular}{lrr" + "rrrrr" * 2 + "}", "\\toprule",
                 "& & & \\multicolumn{5}{c}{no turbulence} & \\multicolumn{5}{c}{moderate turbulence} \\\\", "\\cmidrule(lr){4-8} \\cmidrule(lr){9-13}",
                 "fault (commit) & aircraft & pairs & effect & gated & R & gated R & AUC & effect & gated & R & gated R & AUC \\\\", "\\midrule"]
        for v, label in lab5.items():
            if v == "B_turbine_windmill_2db8e408:engine_restart":
                lines.append("\\midrule")
            cells = [label]
            ref = b.get("still", b.get("turb"))
            cells += [str(int(ref.loc[v].n_aircraft)), str(int(ref.loc[v].n_pairs))] if v in ref.index else ["--", "--"]
            for c in ("still", "turb"):
                if c in b and v in b[c].index:
                    r = b[c].loc[v]; cells += [f"{r.es:.2f}", f"{r.gated:.2f}", f2(r.r), f2(r.gr), f2(r.auc)]
                else:
                    cells += ["--"] * 5
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(TAB, "tab_rq5_conditions.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    # Tier B units (16) x conditions
    modes = [("det_whole", "detector, whole run"), ("det_win30", "detector, 30 s windows"), ("det_win120", "detector, 120 s windows"),
             ("det_phase_any", "detector, any phase (union)"), ("det_phase_max", "detector, phase maximum"),
             ("sig_dynamic_whole", "dynamic signatures, whole run"), ("sig_dynamic_phase_any", "dynamic signatures, any phase"),
             ("envelope_whole", "envelope, whole run")]
    t = {}
    for c in COND:
        pth = os.path.join(COND[c], "16_tier_b.csv")
        if os.path.exists(pth):
            d = pd.read_csv(pth); d = d[d["mode"] != "mode"]
            for col in ("recall", "precision", "gated_recall", "false_alarm"):
                d[col] = pd.to_numeric(d[col], errors="coerce")
            f = d[d.variant != "noop"].groupby("mode")[["recall", "precision", "gated_recall"]].mean()
            fa = d[d.variant == "noop"].groupby("mode")["false_alarm"].mean()
            t[c] = (f, fa)
    if t:
        lines = ["\\begin{tabular}{l" + "cccc" * 2 + "}", "\\toprule", "& \\multicolumn{4}{c}{no turbulence} & \\multicolumn{4}{c}{moderate turbulence} \\\\",
                 "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}", "scoring & R & P & gated R & FA & R & P & gated R & FA \\\\", "\\midrule"]
        for m, label in modes:
            cells = [label]
            for c in ("still", "turb"):
                if c in t and m in t[c][0].index:
                    r = t[c][0].loc[m]; cells += [f2(r.recall), f2(r.precision), f2(r.gated_recall), (f"{100 * t[c][1][m]:.1f}\\%" if m in t[c][1].index else "--")]
                else:
                    cells += ["--"] * 4
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(TAB, "tab_rq4_conditions.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "..", "results", "conditions.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))
    print(json.dumps(res, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))[:5000])


if __name__ == "__main__":
    main()

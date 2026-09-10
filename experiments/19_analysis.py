"""Post-hoc analyses on the per-output records of 08 (Tier A, 25 aircraft), 09 and 16:
  - calibration: false alarms per output, per unchanged test case and per unchanged version (no-op variant);
  - ranking measures per faulty test case: recall@3, recall@5, average precision, outputs inspected before the
    first affected one, share of test cases with at least one correct flag, reduction of the list;
  - AUC by effect-size bin per method (why the residual and the envelope rank differently under the gate);
  - paired hierarchical bootstrap confidence intervals (resample aircraft, then manoeuvres within aircraft)
    for recall/precision/AUC per method, residual minus envelope, dynamic minus static, phase minus whole run.
Writes results/19_analysis.json and paper/tables/tab_ci.tex."""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"))
TAB = os.path.join(HERE, "..", os.environ["DEVSIG_RESULTS"], "tables") if "DEVSIG_RESULTS" in os.environ else os.path.join(HERE, "..", "paper", "tables")
N_BOOT = 2000
METHODS = ["regression", "envelope", "sig_dynamic", "sig_static", "ks", "iforest"]
MLABEL = {"regression": "regression residual", "envelope": "envelope", "sig_dynamic": "dynamic signatures",
          "sig_static": "static signatures", "ks": "KS test", "iforest": "isolation forest"}


def hier_boot(df, stat, n_boot=N_BOOT, seed=0):
    """Hierarchical bootstrap of a mean: resample aircraft with replacement, then manoeuvres within each aircraft.
    `stat` is either a column name (mean of that column) or a function of a row-level DataFrame returning the
    per-row quantity as a Series (used for paired differences). Vectorised: rows are first averaged per
    (aircraft, manoeuvre). Returns (point, lo, hi) with 2.5 and 97.5 percentiles."""
    vals = df[stat] if isinstance(stat, str) else stat(df)
    g = pd.DataFrame({"aircraft": df["aircraft"].to_numpy(), "manoeuvre": df["manoeuvre"].to_numpy(), "v": np.asarray(vals, dtype=float)})
    g = g.dropna().groupby(["aircraft", "manoeuvre"])["v"].mean().reset_index()
    per_ac = [grp["v"].to_numpy() for _, grp in g.groupby("aircraft")]
    if not per_ac:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    M = np.stack([v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1) for v in per_ac])  # (n_ac, n_boot)
    pick = rng.integers(0, len(per_ac), size=(n_boot, len(per_ac)))
    boots = M[pick, np.arange(n_boot)[:, None]].mean(axis=1)
    point = float(np.mean([v.mean() for v in per_ac]))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def ranking_measures(po):
    """Per faulty test case and method: rank outputs by score relative to threshold; measures against the oracle."""
    rows = []
    for (ac, man, meth, f), g in po[po.fault != "noop"].groupby(["aircraft", "manoeuvre", "method", "fault"]):
        g = g.sort_values("rsd", ascending=False)
        rel = g["in_oracle"].astype(bool).to_numpy()
        n_pos = int(rel.sum())
        if n_pos == 0:
            continue
        hits = np.cumsum(rel)
        prec_at = hits / np.arange(1, len(rel) + 1)
        ap = float((prec_at * rel).sum() / n_pos)
        first = int(np.argmax(rel)) + 1
        flagged = g[g.flag]
        rows.append(dict(aircraft=ac, manoeuvre=man, method=meth, fault=f, n_outputs=len(g), n_oracle=n_pos,
                         recall_at3=float(rel[:3].sum() / n_pos), recall_at5=float(rel[:5].sum() / n_pos), ap=ap,
                         inspect_before_first=first, top_is_affected=bool(rel[0]), n_flagged=len(flagged),
                         any_correct_flag=bool(flagged["in_oracle"].astype(bool).any()),
                         reduction=float(1 - len(flagged) / len(g))))
    return pd.DataFrame(rows)


def main():
    po = pd.read_csv(os.path.join(RES, "08_per_output.csv"))
    r08 = pd.read_csv(os.path.join(RES, "08_baselines_rq2.csv"))
    out = {}
    # --- calibration on the no-op variant
    noop = po[po.fault == "noop"]
    cal = {}
    for m in METHODS:
        x = noop[noop.method == m]
        per_case = x.groupby(["aircraft", "manoeuvre"])["flag"].any()
        per_version = x.groupby("aircraft")["flag"].any()
        cal[m] = dict(false_alarm_per_output=round(float(x.flag.mean()), 4),
                      p_unchanged_test_case_flagged=round(float(per_case.mean()), 3),
                      p_unchanged_version_flagged=round(float(per_version.mean()), 3),
                      flags_per_unchanged_test_case=round(float(x.groupby(["aircraft", "manoeuvre"])["flag"].sum().mean()), 2))
    out["calibration"] = cal
    # the same decision at a stricter level (alpha / 22 outputs: family-wise control per test case), from the p-values
    strict = {}
    if "p" in po:
        a_strict = 0.05 / 22
        for m in METHODS:
            x = noop[noop.method == m]; f = po[(po.fault != "noop") & (po.method == m)]
            fl = x.p < a_strict; fl_f = f.p < a_strict
            strict[m] = dict(alpha=a_strict, false_alarm_per_output=round(float(fl.mean()), 4),
                             p_unchanged_test_case_flagged=round(float(fl.groupby([x.aircraft, x.manoeuvre]).any().mean()), 3),
                             p_unchanged_version_flagged=round(float(fl.groupby(x.aircraft).any().mean()), 3),
                             recall=round(float((fl_f & f.in_oracle.astype(bool)).sum() / f.in_oracle.astype(bool).sum()), 3),
                             precision=round(float((fl_f & f.in_oracle.astype(bool)).sum() / max(1, fl_f.sum())), 3))
    out["calibration_strict"] = strict
    # version-level decisions from the stored p-values: one candidate version = all its test cases on one aircraft
    # (about 22 outputs x 9 battery manoeuvres). Bonferroni over that family, and Benjamini-Hochberg FDR per version.
    def bh(pv, q):
        pv = np.asarray(pv, dtype=float); n = len(pv); o = np.argsort(pv); ps = pv[o]
        k = np.where(ps <= q * np.arange(1, n + 1) / n)[0]; keep = np.zeros(n, bool)
        if len(k):
            keep[o[:k.max() + 1]] = True
        return keep
    def by(pv, q):  # Benjamini-Yekutieli: valid under arbitrary dependence among the tests
        pv = np.asarray(pv, dtype=float); n = len(pv); o = np.argsort(pv); ps = pv[o]; c = float(np.sum(1.0 / np.arange(1, n + 1)))
        k = np.where(ps <= q * np.arange(1, n + 1) / (n * c))[0]; keep = np.zeros(n, bool)
        if len(k):
            keep[o[:k.max() + 1]] = True
        return keep
    from scipy.stats import beta as _beta
    def cp(k, n, a=0.05):  # Clopper-Pearson interval for k of n
        lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1)); hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
        return round(lo, 3), round(hi, 3)
    rules = {"per_output_0.05": lambda g: (g.p < 0.05).to_numpy(), "bonferroni_version": lambda g: (g.p < 0.05 / len(g)).to_numpy(),
             "fdr_0.05_version": lambda g: bh(g.p, 0.05), "fdr_0.10_version": lambda g: bh(g.p, 0.10), "by_0.05_version": lambda g: by(g.p, 0.05)}
    version_level = {}
    if "p" in po:
        for m in METHODS:
            x = noop[noop.method == m]; f = po[(po.fault != "noop") & (po.method == m)]
            version_level[m] = {}
            for name, rule in rules.items():
                fl = pd.Series(np.concatenate([rule(g) for _, g in x.groupby("aircraft")]), index=pd.concat([g for _, g in x.groupby("aircraft")]).index)
                ff = pd.Series(np.concatenate([rule(g) for _, g in f.groupby(["aircraft", "fault"])]), index=pd.concat([g for _, g in f.groupby(["aircraft", "fault"])]).index)
                orc = f.in_oracle.astype(bool).reindex(ff.index); gated = (f.es > 1).reindex(ff.index) & orc
                ver = fl.groupby(x.aircraft.reindex(fl.index)).any(); kv, nv = int(ver.sum()), int(len(ver))
                per_case = f.assign(_fl=ff.reindex(f.index).values).groupby(["aircraft", "manoeuvre", "fault"]).apply(
                    lambda t: pd.Series(dict(any_correct=bool((t._fl & t.in_oracle.astype(bool)).any()), reduction=1 - t._fl.mean())))
                ni = (f.fault != "inertia_x1.6").reindex(ff.index)  # the inertia oracle covers every output: trivially correct flags
                pc_ni = per_case.reset_index(); pc_ni = pc_ni[pc_ni.fault != "inertia_x1.6"]
                version_level[m][name] = dict(unchanged_versions_flagged=f"{kv}/{nv}", p_unchanged_version_ci95=cp(kv, nv),
                                              precision_no_inertia=round(float((ff & orc & ni).sum() / max(1, (ff & ni).sum())), 3),
                                              oracle_share=round(float(orc.mean()), 3), oracle_share_no_inertia=round(float(orc[ni].mean()), 3),
                                              any_correct_flag_no_inertia=round(float(pc_ni.any_correct.mean()), 3), reduction_no_inertia=round(float(pc_ni.reduction.mean()), 3),
                                              gated_precision=round(float((ff & gated).sum() / max(1, ff.sum())), 3),
                                              share_flags_on_oracle_below_gate=round(float((ff & orc & ~gated).sum() / max(1, ff.sum())), 3),
                                              share_off_oracle_flags_moved=round(float((ff & ~orc & (f.es.reindex(ff.index) > 1)).sum() / max(1, (ff & ~orc).sum())), 3),
                                              any_correct_flag=round(float(per_case.any_correct.mean()), 3), reduction=round(float(per_case.reduction.mean()), 3),
                                              p_unchanged_version_flagged=round(float(ver.mean()), 3),
                                              false_flags_per_version=round(float(fl.groupby(x.aircraft.reindex(fl.index)).sum().mean()), 2),
                                              p_unchanged_test_case_flagged=round(float(fl.groupby([x.aircraft.reindex(fl.index), x.manoeuvre.reindex(fl.index)]).any().mean()), 3),
                                              recall=round(float((ff & orc).sum() / orc.sum()), 3), precision=round(float((ff & orc).sum() / max(1, ff.sum())), 3),
                                              gated_recall=round(float((ff & gated).sum() / max(1, gated.sum())), 3))
    out["version_level"] = version_level
    # --- ranking measures
    rk = ranking_measures(po)
    rk.to_csv(os.path.join(RES, "19_ranking.csv"), index=False)
    out["ranking"] = {m: {k: round(float(v), 3) for k, v in rk[rk.method == m][["recall_at3", "recall_at5", "ap", "inspect_before_first", "any_correct_flag", "reduction"]].mean().items()}
                      for m in METHODS}
    rkn = rk[rk.fault != "inertia_x1.6"]  # the inertia oracle covers every output, so every ordering is perfect there
    out["ranking_no_inertia"] = {m: {k: round(float(v), 3) for k, v in rkn[rkn.method == m][["ap", "inspect_before_first", "any_correct_flag", "top_is_affected"]].mean().items()}
                                  for m in METHODS}
    def chance(r):
        """Expected values under a random ordering of the outputs of each test case: first affected rank (n+1)/(k+1),
        share with an affected output first k/n, and average precision (exact expectation, by simulation per (k, n))."""
        rng = np.random.default_rng(0); cache = {}
        def eap(k, n):
            if (k, n) not in cache:
                rel = np.zeros((4000, n), bool)
                for i in range(4000):
                    rel[i, rng.choice(n, k, replace=False)] = True
                hits = np.cumsum(rel, axis=1); cache[(k, n)] = float(((hits / np.arange(1, n + 1)) * rel).sum(axis=1).mean() / k)
            return cache[(k, n)]
        x = r[r.method == "regression"]
        return dict(first_affected_rank=round(float(((x.n_outputs + 1) / (x.n_oracle + 1)).mean()), 3), top_is_affected=round(float((x.n_oracle / x.n_outputs).mean()), 3),
                    ap=round(float(np.mean([eap(int(k), int(n)) for k, n in zip(x.n_oracle, x.n_outputs)])), 3))
    out["ranking_chance"] = dict(all=chance(rk), no_inertia=chance(rkn))
    # --- AUC by effect-size bin per method (faulty test cases, outputs with a defined effect size)
    fx = po[(po.fault != "noop") & po.es.notna()].copy()
    fx["es_bin"] = pd.cut(fx.es, [0, 1, 2, 4, 8, np.inf], labels=["<1", "1-2", "2-4", "4-8", ">8"], right=False)
    flag_rate = fx.groupby(["method", "es_bin"], observed=True)["flag"].mean().unstack()
    out["flag_rate_by_effect_size"] = {m: {str(k): round(float(v), 3) for k, v in flag_rate.loc[m].items()} for m in METHODS if m in flag_rate.index}
    in_oracle_by_bin = fx.groupby("es_bin", observed=True)["in_oracle"].mean()
    out["oracle_share_by_effect_size"] = {str(k): round(float(v), 3) for k, v in in_oracle_by_bin.items()}
    # --- bootstrap confidence intervals
    # intervals on the population of the detector table: the nine-manoeuvre battery, aircraft-manoeuvre pairs
    sub = r08[(r08.fault != "noop") & r08.manoeuvre.isin(["cruise", "throttle_step", "elev_doublet", "elev_step", "aileron_step", "rudder_doublet", "bank_hold", "flap_extend", "engine_cut"])]
    ci = {}
    for m in METHODS:
        x = sub[sub.method == m]
        for k in ("recall", "precision", "auc", "gated_recall", "gated_auc"):
            if k in x:
                ci[f"{m}:{k}"] = hier_boot(x, k)
    # paired differences: residual minus envelope on the same test cases
    piv = sub.pivot_table(index=["aircraft", "manoeuvre", "fault"], columns="method", values=["recall", "precision", "auc", "gated_recall", "gated_auc"]).reset_index()
    piv.columns = ["_".join(c).strip("_") for c in piv.columns]
    for other in [m for m in METHODS if m != "regression"]:
        tag = "envelope" if other == "envelope" else other
        for k in ("recall", "precision", "auc", "gated_recall", "gated_auc"):
            a, b = f"{k}_regression", f"{k}_{other}"
            if a in piv and b in piv:
                d = piv.dropna(subset=[a, b])
                ci[f"residual_minus_{tag}:{k}"] = hier_boot(d, lambda z, a=a, b=b: z[a] - z[b])
    p09 = os.path.join(RES, "09_explain_rq2b.csv")
    if os.path.exists(p09):
        e = pd.read_csv(p09)
        pv = e.pivot_table(index=["aircraft", "manoeuvre", "fault"], columns="explainer", values=["agreement", "correctness"]).reset_index()
        pv.columns = ["_".join(c).strip("_") for c in pv.columns]
        for k in ("agreement", "correctness"):
            d = pv.dropna(subset=[f"{k}_dynamic", f"{k}_static"])
            ci[f"dynamic_minus_static:{k}"] = hier_boot(d, lambda z, k=k: z[f"{k}_dynamic"] - z[f"{k}_static"])
            ci[f"dynamic:{k}"] = hier_boot(e[e.explainer == "dynamic"], k)
    p16 = os.path.join(RES, "16_tier_b.csv")
    if os.path.exists(p16):
        t = pd.read_csv(p16); t = t[t.variant != "noop"].rename(columns={"mission": "manoeuvre"})
        for a, b in (("det_phase_any", "det_whole"), ("det_phase_max", "det_whole")):
            pv = t[t["mode"].isin([a, b])].pivot_table(index=["aircraft", "manoeuvre", "variant"], columns="mode", values="recall").dropna().reset_index()
            if len(pv):
                ci[f"{a}_minus_{b}:recall"] = hier_boot(pv, lambda z, a=a, b=b: z[a] - z[b])
    out["ci"] = {k: [round(v, 3) for v in vals] for k, vals in ci.items()}
    # matched replay: compare each candidate run with the baseline run of the same seed (identical inputs, wind, load
    # and gusts). The effect size es is that matched distance over the benign variation, so "es > 0" flags any change
    # and "es > 1" flags a change beyond the benign variation. Scored against the physics-derived oracle only: the gated
    # oracle is built from the same matched distance and would score it against itself.
    try:
        from sklearn.metrics import roc_auc_score
        q9 = po[po.manoeuvre.isin(["cruise", "throttle_step", "elev_doublet", "elev_step", "aileron_step", "rudder_doublet", "bank_hold", "flap_extend", "engine_cut"])
                & (po.method == "regression")]
        f9, n9 = q9[q9.fault != "noop"], q9[q9.fault == "noop"]
        orc9 = f9.in_oracle.astype(bool)
        mr = {}
        for name, thr in (("any_difference", 1e-9), ("beyond_benign_spread", 1.0)):
            # recall and precision per test case, then averaged, as for the other detectors (precision undefined when
            # nothing is flagged)
            rec, prec = [], []
            for _, g in f9.groupby(["aircraft", "manoeuvre", "fault"]):
                o = g.in_oracle.astype(bool); fl = g.es > thr
                if o.sum():
                    rec.append((fl & o).sum() / o.sum())
                if fl.sum() and g.fault.iloc[0] != "inertia_x1.6":  # precision without the trivially correct inertia instance
                    prec.append((fl & o).sum() / fl.sum())
            mr[name] = dict(recall=round(float(np.mean(rec)), 3), precision=round(float(np.mean(prec)), 3),
                            noop_false_alarm=round(float((n9.es > thr).mean()), 4))
        ni9 = f9.fault != "inertia_x1.6"
        mr["outside_oracle_share_of_changed"] = round(float(((f9.es > 1e-9) & ~orc9 & ni9).sum() / max(1, ((f9.es > 1e-9) & ni9).sum())), 3)
        rk = {"residual": [], "matched": []}
        for _, g in f9.groupby(["aircraft", "manoeuvre", "fault"]):
            y = g.in_oracle.astype(bool).to_numpy()
            if y.all() or not y.any():
                continue
            for k, col in (("residual", "rsd"), ("matched", "es")):
                sc = g[col].to_numpy(); o = np.argsort(-sc)
                rk[k].append((roc_auc_score(y, sc), int(np.argmax(y[o])) + 1, bool(y[o][0])))
        for k, v in rk.items():
            a = np.array(v, dtype=float)
            mr[f"ranking_{k}"] = dict(auc=round(float(a[:, 0].mean()), 3), first_affected_rank=round(float(a[:, 1].mean()), 2), top_is_affected=round(float(a[:, 2].mean()), 3), n=int(len(a)))
        out["matched_replay"] = mr
    except Exception as e:  # noqa: BLE001
        out["matched_replay_error"] = repr(e)
    # run validity: a candidate run that does not complete where the matched baseline run completed is a structural
    # regression, reported separately from the behavioural decision
    import glob
    runs = os.path.join(RES, "runs", "tier_a"); struct = {}
    variants = ["noop", "clo_+0.1", "cmalpha_x0.7", "cmq_x0.5", "inertia_x1.6", "B_elevator_drag_8819410c", "B_pqrdot_5ad2694c",
                "B_piston_power_bcd3f980", "B_turbine_windmill_2db8e408", "B_ground_effect_9c058118"]
    for acd in glob.glob(os.path.join(runs, "*")):
        files = set(os.listdir(acd))
        for fn in files:
            if not fn.startswith("baseline_"):
                continue
            man, seed = fn[len("baseline_"):].rsplit("_", 1); seed = int(seed.split(".")[0])
            if seed < 100:
                continue
            for v in variants:
                if not any(x.startswith(f"{v}_{man}_") for x in files):
                    continue
                d = struct.setdefault(v, dict(matched_baseline_runs=0, candidate_runs_missing=0, test_cases=set(), test_cases_with_missing=set()))
                d["matched_baseline_runs"] += 1; d["test_cases"].add((os.path.basename(acd), man))
                if f"{v}_{man}_{seed:04d}.parquet" not in files:
                    d["candidate_runs_missing"] += 1; d["test_cases_with_missing"].add((os.path.basename(acd), man))
    out["structural_failures"] = {v: dict(matched_baseline_runs=d["matched_baseline_runs"], candidate_runs_missing=d["candidate_runs_missing"],
                                          test_cases=len(d["test_cases"]), test_cases_with_missing=len(d["test_cases_with_missing"])) for v, d in struct.items()}
    with open(os.path.join(RES, "19_analysis.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    # LaTeX table of the principal intervals
    os.makedirs(TAB, exist_ok=True)
    lines = ["\\begin{tabular}{lccc}", "\\toprule", "quantity & recall & precision & AUC \\\\", "\\midrule"]
    fmt = lambda k: (f"{ci[k][0]:.2f} [{ci[k][1]:.2f}, {ci[k][2]:.2f}]" if k in ci else "--")
    for m in METHODS:
        lines.append(f"{MLABEL[m]} & {fmt(f'{m}:recall')} & {fmt(f'{m}:precision')} & {fmt(f'{m}:auc')} \\\\")
    lines += ["\\midrule", f"residual $-$ envelope & {fmt('residual_minus_envelope:recall')} & -- & {fmt('residual_minus_envelope:auc')} \\\\",
              f"residual $-$ envelope, gated & {fmt('residual_minus_envelope:gated_recall')} & -- & {fmt('residual_minus_envelope:gated_auc')} \\\\"]
    if "dynamic_minus_static:correctness" in ci:
        lines.append(f"dynamic $-$ static rules (agreement / correctness) & {fmt('dynamic_minus_static:agreement')} & {fmt('dynamic_minus_static:correctness')} & -- \\\\")
    for k in ("det_phase_any_minus_det_whole:recall", "det_phase_max_minus_det_whole:recall"):
        if k in ci:
            lines.append(f"{k.split(':')[0].replace('_', ' ')} & {fmt(k)} & -- & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "tab_ci.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()

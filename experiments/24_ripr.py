"""Revelation stages and pre-check signals for the real faults (protocol/real_fault_protocol.md, Sections 4-5).
Inputs: <R>/23_real_fault_cases.csv and 23_real_fault_outputs.csv for R in results_still (no turbulence) and results
(moderate turbulence); results/25_coverage_reach.csv (reach, measured without turbulence, used for both conditions).
Unit: one real fault on one aircraft in one test case (a candidate version = one fault on one aircraft, over its
test cases). Revealed: structural regression, or the operational decision (Benjamini-Hochberg q = 0.05 over all
outputs and test cases of the version) flags an output whose effect size exceeds 1.
Writes results/24_ripr.json, results/24_ripr_units.csv, paper/tables/tab_ripr_stages.tex, tab_ripr_signals.tex."""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
COND = {"no turbulence": os.path.join(REPO, "results_still"), "moderate turbulence": os.path.join(REPO, "results")}
QTG = ["cruise", "throttle_step", "elev_doublet", "elev_step", "aileron_step", "rudder_doublet", "bank_hold", "flap_extend", "engine_cut"]
TAB = os.path.join(REPO, "paper", "tables")
LABEL = {"B_elevator_drag_8819410c": "elevator drag", "B_ground_effect_9c058118": "ground-effect table", "B_pqrdot_5ad2694c": "angular acceleration",
         "B_piston_power_bcd3f980": "piston power, engine off", "B_turbine_windmill_2db8e408": "turbine windmilling",
         "B2_turbine_throttle_init_c8af9244": "turbine throttle initialisation", "B2_turbine_thruster_inputs_20066483": "turbine thruster inputs",
         "B2_wheel_spin_down_8a9fba1c": "wheel spin-down", "B2_sensor_fail_stuck_f543d634": "sensor stuck failure",
         "B2_atmosphere_high_altitude_61f1e8c3": "atmosphere above 91 km", "B2_load_factor_sign_c3fa7d22": "load-factor sign",
         "B2_propeller_inflow_angle_82cc3893": "propeller inflow angle", "B2_atmosphere_gradient_fadeout_05b709c8": "temperature-gradient fade-out",
         "B2_A4_gear_retractable_3be71976_as2007": "A-4 gear retraction flag", "B2_Boeing314_aero_props_75b9b1e8": "Boeing 314 aerodynamic inputs",
         "B2_holdback_thrust_prop_5306a18b": "holdback thrust property", "B2_systems_xml_validation_95e5a16b": "system-file validation"}


# the literal reversal of the A-4 fix on 1.3.1 fails to load because the newer parser rejects the text flag; the 2007
# parser read it as 0 (gear not retractable), reproduced by the _as2007 variant, which replaces it in every count
LITERAL_A4 = "B2_A4_gear_retractable_3be71976"


def bh(p, q=0.05):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); k = np.where(p[o] <= q * np.arange(1, n + 1) / n)[0]
    keep = np.zeros(n, bool)
    if len(k):
        keep[o[:k.max() + 1]] = True
    return keep


def units(cond_dir, reach):
    import sys
    sys.path.insert(0, HERE)
    applies = {k: set(v[2]) for k, v in __import__("23_real_fault_scores").FAULTS.items()}
    cases = pd.read_csv(os.path.join(cond_dir, "23_real_fault_cases.csv"))
    outs = pd.read_csv(os.path.join(cond_dir, "23_real_fault_outputs.csv"))
    keep = lambda d: d[[a in applies.get(f, {a}) for f, a in zip(d.fault, d.aircraft)]]
    cases, outs = keep(cases), keep(outs)
    outs["flag"] = False
    for _, g in outs.groupby(["fault", "aircraft"]):
        outs.loc[g.index, "flag"] = bh(g.p.to_numpy())
    hit = outs[outs.flag & (outs.es > 1)].groupby(["fault", "aircraft", "test_case"]).size().rename("n_correct_flags")
    hit_small = outs[outs.flag & (outs.es > 1e-9)].groupby(["fault", "aircraft", "test_case"]).size().rename("n_flags_on_changed")
    anyflag = outs[outs.flag].groupby(["fault", "aircraft", "test_case"]).size().rename("n_flags")
    u = cases.merge(hit, on=["fault", "aircraft", "test_case"], how="left").merge(anyflag, on=["fault", "aircraft", "test_case"], how="left").merge(hit_small, on=["fault", "aircraft", "test_case"], how="left")
    u[["n_correct_flags", "n_flags", "n_flags_on_changed"]] = u[["n_correct_flags", "n_flags", "n_flags_on_changed"]].fillna(0).astype(int)
    u = u.merge(reach[["fault", "aircraft", "test_case", "reach"]], on=["fault", "aircraft", "test_case"], how="left")
    u["reach"] = np.where(u.kind == "model", True, u.reach.fillna(False).astype(bool))  # model tables are evaluated every frame
    u["propagated"] = u.propagated.fillna(False).astype(bool) | u.structural
    u["observable"] = u.observable.fillna(False).astype(bool) | u.structural
    u["revealed"] = u.structural | (u.n_correct_flags > 0)  # a flag on an output changed beyond the benign variation
    u["revealed_any_change"] = u.structural | (u.n_flags_on_changed > 0)  # a flag on any output the fault changed
    u["flagged_without_change"] = (u.n_flags > 0) & (u.n_flags_on_changed == 0) & ~u.structural  # chance flags
    u["qtg"] = u.test_case.isin(QTG)
    u["signal_es1"] = (u.es1_max.fillna(0) > 1) | u.structural
    return u


def signal_table_behavioural(u, target="revealed"):
    """Signals evaluated on units of faults that leave the model running (structural faults excluded), against the
    strict (revealed) or the broad (revealed_any_change) count of revelation."""
    sf = u.groupby("fault").structural.mean()
    x = u[u.fault.map(sf) < 0.5]
    out = {}
    for name, col in (("QTG-style battery", "qtg"), ("line coverage of the changed code", "reach"),
                      ("one matched pair differs", "propagated"), ("one matched pair beyond the benign variation", "signal_es1")):
        sel = x[col].astype(bool); rev = x[target]
        out[name] = dict(selected=float(sel.mean()), precision=float((sel & rev).sum() / max(1, sel.sum())) if rev.sum() else float("nan"),
                         recall=float((sel & rev).sum() / rev.sum()) if rev.sum() else float("nan"),
                         revealed_outside_selection=int((~sel & rev).sum()))
    out["_n_units"] = int(len(x)); out["_n_revealed"] = int(x[target].sum()); out["_n_faults"] = int(x.fault.nunique())
    return out


def signal_table(u):
    rows = {}
    for name, col in (("QTG-style battery", "qtg"), ("line coverage of the changed code", "reach"),
                      ("one matched pair differs", "propagated"), ("one matched pair beyond the benign variation", "signal_es1")):
        sel = u[col].astype(bool); rev = u.revealed
        rows[name] = dict(selected=float(sel.mean()), precision=float((sel & rev).sum() / max(1, sel.sum())),
                          recall=float((sel & rev).sum() / max(1, rev.sum())), n_selected=int(sel.sum()))
    rows["_n_units"] = int(len(u)); rows["_n_revealed"] = int(u.revealed.sum())
    return rows


def main():
    reach = pd.read_csv(os.path.join(REPO, "results", "25_coverage_reach.csv"))
    out, allu = {}, []
    for cname, cdir in COND.items():
        u = units(cdir, reach); u["condition"] = cname
        lit = u[u.fault == LITERAL_A4]; u = u[u.fault != LITERAL_A4]; allu.append(u)
        per_fault = u.groupby(["set", "fault", "kind"]).agg(aircraft=("aircraft", "nunique"), cases=("test_case", "size"), reach=("reach", "mean"),
                                                          propagated=("propagated", "mean"), observable=("observable", "mean"), revealed=("revealed", "mean"),
                                                          structural=("structural", "sum"), revealed_any_change=("revealed_any_change", "mean"),
                                                          n_revealed=("revealed", "sum"), n_revealed_any_change=("revealed_any_change", "sum"),
                                                          n_flagged_without_change=("flagged_without_change", "sum"),
                                                          n_reach=("reach", "sum"), n_propagated=("propagated", "sum"), n_observable=("observable", "sum")).reset_index()
        by_qtg = u[u.qtg].groupby("fault").revealed.any().rename("revealed_by_qtg"); by_all = u.groupby("fault").revealed.any().rename("revealed_by_suite")
        by_qtg_b = u[u.qtg].groupby("fault").revealed_any_change.any().rename("revealed_by_qtg_broad")
        by_all_b = u.groupby("fault").revealed_any_change.any().rename("revealed_by_suite_broad")
        per_fault = (per_fault.merge(by_qtg, on="fault", how="left").merge(by_all, on="fault", how="left")
                     .merge(by_qtg_b, on="fault", how="left").merge(by_all_b, on="fault", how="left"))
        out[cname] = dict(per_fault=per_fault.to_dict(orient="records"),
                          signals={s: signal_table(u[u.set == s]) for s in ("development", "heldout")},
                          signals_behavioural={s: signal_table_behavioural(u[u.set == s]) for s in ("development", "heldout")},
                          signals_behavioural_broad={s: signal_table_behavioural(u[u.set == s], "revealed_any_change") for s in ("development", "heldout")},
                          revealed_by_qtg_broad=int(u[u.qtg].groupby("fault").revealed_any_change.any().sum()),
                          revealed_by_suite_broad=int(u.groupby("fault").revealed_any_change.any().sum()),
                          protocol_broad_cases=u[(u.set == "heldout") & u.revealed_any_change & ~u.structural][["fault", "aircraft", "test_case"]].to_dict(orient="records"),
                          revealed_by_qtg=int(u[u.qtg].groupby("fault").revealed.any().sum()), revealed_by_suite=int(u.groupby("fault").revealed.any().sum()),
                          n_faults=int(u.fault.nunique()),
                          a4_literal_reversal=dict(cases=int(len(lit)), structural=int(lit.structural.sum())),
                          chance_flagged_cases_per_fault=u[~u.structural].groupby("fault").flagged_without_change.sum().to_dict(),
                          checks=dict(propagated_without_reach_cpp=int(((u.kind == "cpp") & u.propagated & ~u.reach & ~u.structural).sum()),
                                      revealed_without_propagation=int((u.revealed & ~u.propagated).sum()),
                                      reached_but_inert_share_cpp=float(((u.kind == "cpp") & u.reach & ~u.propagated).sum() / max(1, ((u.kind == "cpp") & u.reach).sum()))))
    allu = pd.concat(allu); allu.to_csv(os.path.join(REPO, "results", "24_ripr_units.csv"), index=False)
    with open(os.path.join(REPO, "results", "24_ripr.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else float(o) if isinstance(o, (np.floating,)) else int(o))
    # table: stages per fault, both conditions
    os.makedirs(TAB, exist_ok=True)
    pf = {c: pd.DataFrame(out[c]["per_fault"]).set_index("fault") for c in COND}
    f2 = lambda x: f"{x:.2f}"
    yn = lambda q, a: ("yes" if q else "no") + " / " + ("yes" if a else "no")
    lines = [r"\begin{tabular}{lrr@{\hspace{1.2em}}rrrr@{\hspace{1.2em}}rrrr@{\hspace{1.2em}}l}", "\\toprule",
             "& & & \\multicolumn{4}{c}{no turbulence} & \\multicolumn{4}{c}{moderate turbulence} & \\\\",
             "\\cmidrule(lr){4-7} \\cmidrule(lr){8-11}",
             "& & & & & \\multicolumn{2}{c}{revealed} & & & \\multicolumn{2}{c}{revealed} & \\\\",
             "\\cmidrule(lr){6-7} \\cmidrule(lr){10-11}",
             "fault (aircraft) & cases & \\shortstack{code\\\\runs} & \\shortstack{flight\\\\changes} & \\shortstack{beyond\\\\variation} & broad & strict & \\shortstack{flight\\\\changes} & \\shortstack{beyond\\\\variation} & broad & strict & \\shortstack[l]{QTG-style /\\\\all tests} \\\\", "\\midrule"]
    a, b = pf["no turbulence"], pf["moderate turbulence"]
    for s in ("development", "heldout"):
        lines.append("\\multicolumn{12}{l}{\\emph{" + ("development faults" if s == "development" else "protocol-selected faults") + "}} \\\\")
        for f in a[a.set == s].sort_values(["n_revealed_any_change", "n_observable", "n_propagated", "n_reach"], ascending=False).index:
            ra, rb = a.loc[f], b.loc[f] if f in b.index else None
            last = yn(ra.revealed_by_qtg, ra.revealed_by_suite)
            if (ra.revealed_by_qtg_broad, ra.revealed_by_suite_broad) != (ra.revealed_by_qtg, ra.revealed_by_suite):
                last += "$^{*}$"  # the broad count differs; the caption says how
            cond = lambda r: " & ".join(str(int(r[k])) for k in ("n_propagated", "n_observable", "n_revealed_any_change", "n_revealed")) if r is not None else " & ".join(["--"] * 4)
            lines.append(f"{LABEL.get(f, f)} ({int(ra.aircraft)}) & {int(ra.cases)} & {int(ra.n_reach)} & {cond(ra)} & {cond(rb)} & {last} \\\\")
        lines.append("\\midrule" if s == "development" else "\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(TAB, "tab_ripr_stages.tex"), "w").write("\n".join(lines) + "\n")
    # table: signals on faults that leave the model running; development set (revealed cases exist) and held-out set
    lines = [r"\begin{tabular}{lccccc@{\hspace{0.6em}}c@{\hspace{1.4em}}ccccc@{\hspace{0.6em}}c}", "\\toprule",
             "& \\multicolumn{6}{c}{no turbulence} & \\multicolumn{6}{c}{moderate turbulence} \\\\",
             "\\cmidrule(lr){2-7} \\cmidrule(lr){8-13}",
             "& \\multicolumn{5}{c}{development} & protocol & \\multicolumn{5}{c}{development} & protocol \\\\",
             "\\cmidrule(lr){2-6} \\cmidrule(lr){7-7} \\cmidrule(lr){8-12} \\cmidrule(lr){13-13}",
             "& & \\multicolumn{2}{c}{strict} & \\multicolumn{2}{c}{broad} & & & \\multicolumn{2}{c}{strict} & \\multicolumn{2}{c}{broad} & \\\\",
             "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){9-10} \\cmidrule(lr){11-12}",
             "signal & selected & P & R & P & R & selected & selected & P & R & P & R & selected \\\\", "\\midrule"]
    for name in ("QTG-style battery", "line coverage of the changed code", "one matched pair differs", "one matched pair beyond the benign variation"):
        cells = [name]
        for c in COND:
            d = out[c]["signals_behavioural"]["development"][name]; db = out[c]["signals_behavioural_broad"]["development"][name]
            h = out[c]["signals_behavioural"]["heldout"][name]
            cells += [f2(d["selected"]), f2(d["precision"]), f2(d["recall"]), f2(db["precision"]), f2(db["recall"]), f2(h["selected"])]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "tab_ripr_signals.tex"), "w").write("\n".join(lines) + "\n")
    for c in COND:
        print("==", c); print(pd.DataFrame(out[c]["per_fault"]).round(2).to_string()); print(json.dumps(out[c]["signals"], indent=1)); print(out[c]["checks"])


if __name__ == "__main__":
    main()

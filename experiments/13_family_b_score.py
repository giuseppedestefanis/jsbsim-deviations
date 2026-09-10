"""RQ5: score the Family B variants (real historical faults) with the detector and the explainers.
For each (aircraft, manoeuvre, B variant) with cached runs: effect-size gate vs matched-seed baselines,
detector flags and AUC against the diff-derived oracle, dynamic/static explainer agreement and correctness."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import importlib
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
b6 = importlib.import_module("06_tier_a_battery")
b7 = importlib.import_module("07_dynamic_signatures")
from devsig.manoeuvres import BATTERY  # noqa: E402
from devsig.signature import Signatures  # noqa: E402
from devsig.baselines import RegressionResidual  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402
from devsig.evaluate import prf, to_category, auc  # noqa: E402
from devsig.explain import attribute, correctness, agreement  # noqa: E402

RUNS = b6.RUNS
LONG, LAT = set(b6.LONG), set(b6.LAT)
# oracle from the diff: which outputs the changed term feeds (direct or first-order)
ORACLE_B = {
    "B_elevator_drag_8819410c": LONG,                      # drag term wrong -> speed, altitude, pitch chain
    "B_ground_effect_9c058118": LONG,                      # lift factor near ground (expected below gate at altitude)
    "B_pqrdot_5ad2694c": LONG | LAT,                       # angular acceleration term (planet rotation): all rates
    "B_piston_power_bcd3f980": LONG,                       # negative power at idle -> thrust, speed, altitude
    "B_turbine_windmill_2db8e408": LONG,                   # engine restart after cut -> thrust, speed, altitude
}


# a model-file fix is reversed only on the aircraft whose files the fix changed (8819410c changed 24 aircraft files,
# 10 of them among the 25; 9c058118 changed the 737 and the C172X): reversing it elsewhere would create a fault the
# history never contained
APPLIES = {"B_elevator_drag_8819410c": ["737", "A4", "B747", "F4N", "F80C", "MD11", "OV10", "T37", "T38", "f15"], "B_ground_effect_9c058118": ["737"]}


def variants_present(ac):
    d = os.path.join(RUNS, ac)
    if not os.path.isdir(d):
        return []
    return sorted({f.split("_" + m)[0] for f in os.listdir(d) for m in BATTERY if f.startswith("B_") and ("_" + m + "_") in f})


def one_pair(args):
    ac, man = args
    rows = []
    variants = [v for v in variants_present(ac) if v in ORACLE_B and ac in APPLIES.get(v, [ac])]
    if not variants:
        return rows
    base = b7.load(ac, "baseline", man, range(40)); base_m = b7.load(ac, "baseline", man, range(100, 110))
    if len(base) < 30 or len(base_m) < 8:
        return rows
    train, held = base[:len(base) // 2], base[len(base) // 2:]
    det = RegressionResidual(max_iter=60).fit(train)
    dnull = NullModel(pd.DataFrame([det.score(r) for r in held])); dthr = dnull.threshold("p95")
    sigs = {"dynamic": Signatures(k=5, min_samples_leaf=10, lags_s=(0.5, 1.0, 2.0, 5.0), since_change=True).fit(train),
            "static": Signatures(k=5, min_samples_leaf=10).fit(train)}
    snull = {k: NullModel(pd.DataFrame([s.vsd(r) for r in held])) for k, s in sigs.items()}
    for v in variants:
        cand = b7.load(ac, v, man, range(100, 110))
        if len(cand) < 5:  # pairs with fewer than five surviving candidate runs are skipped (count reported)
            continue
        es = b6.effect_size(cand, base_m, det.outputs)
        orc = pd.Series({m: (m in ORACLE_B[v]) for m in det.outputs})
        gated = orc & (es > 1.0)
        rsd, fl = dnull.flag_perm(pd.DataFrame([det.score(r) for r in cand]))[:2]
        rec = dict(aircraft=ac, manoeuvre=man, variant=v, es_max=float(es.max()), n_gated=int(gated.sum()), n_flagged=int(fl.sum()), n_cand=len(cand))
        for tag, o in (("phys", orc), ("gated", gated)):
            if o.sum() == 0:
                rec[f"{tag}_recall"] = np.nan; rec[f"{tag}_precision"] = np.nan; rec[f"{tag}_auc"] = np.nan; continue
            m = prf(fl, o); c = prf(to_category(fl), to_category(o))
            rec[f"{tag}_recall"] = m["recall"]; rec[f"{tag}_precision"] = m["precision"]
            rec[f"{tag}_cat_recall"] = c["recall"]; rec[f"{tag}_auc"] = auc(rsd, o)
        flagged = list(fl[fl].index)
        for k, s in sigs.items():
            srsd, sfl = snull[k].flag_perm(pd.DataFrame([s.vsd(r) for r in cand]))[:2]
            rec[f"agree_{k}"] = agreement(fl, sfl)
            corr = [c for c in (correctness(attribute(s, cand, o, held), man, o) for o in flagged) if c is not None]
            rec[f"correct_{k}"] = (np.mean(corr) if corr else np.nan)
        rows.append(rec)
    print(ac, man, "done", flush=True)
    return rows


def main():
    cfg = json_aircraft = sorted(os.listdir(RUNS))
    aircraft = sys.argv[1:] or cfg
    pairs = [(ac, man) for ac in aircraft for man in BATTERY]
    with Pool(max(1, min(6, os.cpu_count() - 1))) as pool:
        results = pool.map(one_pair, pairs, chunksize=1)
    out = pd.DataFrame([r for rs in results for r in rs])
    out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "13_family_b.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n== per variant: aircraft, pairs, effect-size gate hit rate, detector recall/precision/AUC (physics and gated oracle), explainer agreement and correctness")
    g = out.groupby("variant").agg(n_aircraft=("aircraft", "nunique"), n_pairs=("manoeuvre", "size"), es_max_median=("es_max", "median"),
                                   frac_gated=("n_gated", lambda x: float((x > 0).mean())),
                                   phys_recall=("phys_recall", "mean"), phys_precision=("phys_precision", "mean"), phys_auc=("phys_auc", "mean"),
                                   gated_recall=("gated_recall", "mean"), gated_auc=("gated_auc", "mean"),
                                   agree_dyn=("agree_dynamic", "mean"), correct_dyn=("correct_dynamic", "mean"), correct_stat=("correct_static", "mean"))
    print(g.round(2).to_string())
    # paper table (RQ5): one row per fault, primary and gated oracle side by side
    label = {"B_elevator_drag_8819410c": "elevator drag (8819410c)", "B_ground_effect_9c058118": "ground-effect kink (9c058118)",
             "B_pqrdot_5ad2694c": "angular acceleration (5ad2694c)", "B_piston_power_bcd3f980": "piston negative power (bcd3f980)",
             "B_turbine_windmill_2db8e408": "turbine windmilling (2db8e408)"}
    lines = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
             "fault (commit) & aircraft & pairs & effect size & gated & \\multicolumn{2}{c}{recall} & \\multicolumn{2}{c}{AUC} & correctness \\\\",
             " & & & & & primary & gated & primary & gated & \\\\", "\\midrule"]
    for v in label:
        if v not in g.index:
            continue
        r = g.loc[v]
        f = lambda x: "--" if pd.isna(x) else f"{x:.2f}"
        lines.append(f"{label[v]} & {int(r.n_aircraft)} & {int(r.n_pairs)} & {r.es_max_median:.2f} & {r.frac_gated:.2f} & {f(r.phys_recall)} & {f(r.gated_recall)} & {f(r.phys_auc)} & {f(r.gated_auc)} & {f(r.correct_dyn)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tab = os.path.join(HERE, "..", os.environ["DEVSIG_RESULTS"], "tables") if "DEVSIG_RESULTS" in os.environ else os.path.join(HERE, "..", "paper", "tables"); os.makedirs(tab, exist_ok=True)
    with open(os.path.join(tab, "tab_rq5.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n== elevator drag by manoeuvre (detector gated recall / AUC):")
    e = out[out.variant == "B_elevator_drag_8819410c"]
    print(e.groupby("manoeuvre")[["es_max", "n_gated", "n_flagged", "gated_recall", "gated_auc", "correct_dynamic"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()

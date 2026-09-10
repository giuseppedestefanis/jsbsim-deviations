"""Score every real fault (development set and held-out set, config/real_faults_heldout.csv) per aircraft and test
case, as defined in protocol/real_fault_protocol.md, Sections 4 and 5:
  structural  candidate runs missing where the matched baseline run completed
  propagated  the seed-100 candidate and baseline runs differ bitwise in any recorded quantity
  es1_max     largest output distance in that one matched pair over the benign spread (baseline runs only)
  es_max      largest effect size from the ten matched pairs (the gate); observable = es_max > 1
  per-output permutation p-values of the residual detector, for the version-level decision in script 24.
Writes <results>/23_real_fault_cases.csv and 23_real_fault_outputs.csv. Honours DEVSIG_RESULTS."""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from devsig.run import load_run  # noqa: E402
from devsig.baselines import RegressionResidual  # noqa: E402
from devsig.manoeuvres import BATTERY  # noqa: E402
from devsig.signature import split_metrics  # noqa: E402
from devsig.threshold import NullModel  # noqa: E402
b6 = __import__("06_tier_a_battery")

RES = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"))
RUNS = os.path.join(RES, "runs", "tier_a")
LOWPASS = os.path.join(RES, "runs", "coverage", "737")
ALL = ["737", "787-8", "A320", "A4", "B747", "Boeing314", "Camel", "F4N", "F80C", "J3Cub", "MD11", "OV10", "Short_S23", "T37", "T38",
       "c172p", "c172r", "c182", "c310", "f15", "f16", "global5000", "pa28", "pc7", "t6texan2"]
PISTON = ["Boeing314", "Camel", "J3Cub", "Short_S23", "c172p", "c172r", "c182", "c310", "pa28"]
JETS = ["737", "787-8", "A320", "A4", "B747", "F4N", "F80C", "MD11", "T37", "T38", "f15", "f16", "global5000"]
ELEV_DRAG = ["737", "A4", "B747", "F4N", "F80C", "MD11", "OV10", "T37", "T38", "f15"]  # the Tier A aircraft whose files 8819410c changed


def faults():
    """label -> (set, kind, aircraft the fault applies to)."""
    f = {"B_elevator_drag_8819410c": ("development", "model", ELEV_DRAG), "B_ground_effect_9c058118": ("development", "model", ["737"]),
         "B_pqrdot_5ad2694c": ("development", "cpp", ALL), "B_piston_power_bcd3f980": ("development", "cpp", PISTON),
         "B_turbine_windmill_2db8e408": ("development", "cpp", JETS)}
    d = pd.read_csv(os.path.join(HERE, "..", "config", "real_faults_heldout.csv")); d = d[d.decision == "keep"]
    for _, r in d.iterrows():
        f[r.label] = ("heldout", r.kind, ALL if r.aircraft == "all" else r.aircraft.split(";"))
    # the A-4 fault as the 2007 parser read it: the text flag RETRACT was converted with atof to 0 (gear not
    # retractable); the 1.3.1 parser rejects the text, so the reversed file alone does not reproduce the 2007 symptom
    f["B2_A4_gear_retractable_3be71976_as2007"] = ("heldout", "model", ["A4"])
    return f


FAULTS = faults()
# ONLY_FAULTS=label[,label...] scores just those faults and merges them into the existing outputs (the scoring of
# one fault on one aircraft and test case does not depend on the other faults)
ONLY = [x for x in os.environ.get("ONLY_FAULTS", "").split(",") if x]
if ONLY:
    FAULTS = {k: v for k, v in FAULTS.items() if k in ONLY}


def path(ac, variant, man, seed):
    if man == "low_pass":
        return os.path.join(LOWPASS, f"{variant}_low_pass_{seed:04d}.parquet")
    return os.path.join(RUNS, ac, f"{variant}_{man}_{seed:04d}.parquet")


def load(ac, variant, man, seeds):
    return {s: load_run(path(ac, variant, man, s)) for s in seeds if os.path.exists(path(ac, variant, man, s))}


def rms(a, b, col):
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[col].to_numpy()[:n] - b[col].to_numpy()[:n]) ** 2)))


def one_pair(args):
    ac, man = args
    base = load(ac, "baseline", man, range(40))
    base_m = load(ac, "baseline", man, range(100, 110))
    if len(base) < 30 or len(base_m) < 8:
        return [], []
    b = [base[s] for s in sorted(base)]
    train, held = b[:20], b[20:]
    _, outputs = split_metrics(train[0].columns)
    det = RegressionResidual(max_iter=60).fit(train)
    null = NullModel(pd.DataFrame([det.score(r) for r in held]))
    bm = [base_m[s] for s in sorted(base_m)]
    spread = {}
    for o in outputs:  # benign spread: consecutive matched baseline runs (the gate's denominator)
        d = [rms(bm[i], bm[i + 1], o) for i in range(len(bm) - 1)]
        pooled = np.concatenate([x[o].to_numpy() for x in bm]).std()
        spread[o] = max(float(np.mean(d)), 1e-3 * float(pooled) + 1e-12)
    cases, outs = [], []
    for label, (fset, kind, acs) in FAULTS.items():
        if ac not in acs:
            continue
        cand = load(ac, label, man, range(100, 110))
        matched = [s for s in base_m]
        missing = [s for s in matched if s not in cand]
        rec = dict(fault=label, set=fset, kind=kind, aircraft=ac, test_case=man, n_matched=len(matched), n_cand=len(cand),
                   n_missing=len(missing), structural=len(missing) > 0)
        s0 = 100 if (100 in cand and 100 in base_m) else next((s for s in sorted(cand) if s in base_m), None)
        if s0 is not None:
            c0, b0 = cand[s0], base_m[s0]
            num = [c for c in b0.columns if c in c0.columns and c != "simulation/sim-time-sec" and np.issubdtype(b0[c].dtype, np.number)]
            n = min(len(c0), len(b0))
            rec["propagated"] = bool(len(c0) != len(b0) or not np.array_equal(c0[num].to_numpy()[:n], b0[num].to_numpy()[:n]))
            es1 = {o: rms(c0, b0, o) / spread[o] for o in outputs}
            rec["es1_max"] = float(max(es1.values())); rec["es1_output"] = max(es1, key=es1.get)
        if len(cand) >= 5:
            cl = [cand[s] for s in sorted(cand)]; bl = [base_m[s] for s in sorted(cand) if s in base_m]
            es = pd.Series({o: float(np.mean([rms(c, bb, o) for c, bb in zip(cl, bl)]) / spread[o]) for o in outputs})
            rsd, fl, p = null.flag_perm(pd.DataFrame([det.score(r) for r in cl]))
            rec.update(es_max=float(es.max()), observable=bool((es > 1).any()), n_observable_outputs=int((es > 1).sum()),
                       n_flag_05=int(fl.sum()), min_p=float(p.min()))
            for o in outputs:
                outs.append(dict(fault=label, set=fset, aircraft=ac, test_case=man, output=o, p=float(p[o]), es=float(es[o]), rsd=float(rsd[o])))
        cases.append(rec)
    print(ac, man, "done", flush=True)
    return cases, outs


def main():
    pairs = [(ac, man) for ac in ALL for man in BATTERY] + [("737", "low_pass")]
    if ONLY:
        acs = set().union(*(set(v[2]) for v in FAULTS.values()))
        pairs = [pr for pr in pairs if pr[0] in acs]
    with Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        res = pool.map(one_pair, pairs, chunksize=1)
    cases = pd.DataFrame([c for cs, _ in res for c in cs]); outs = pd.DataFrame([o for _, os_ in res for o in os_])
    if ONLY:
        old_c = pd.read_csv(os.path.join(RES, "23_real_fault_cases.csv")); old_o = pd.read_csv(os.path.join(RES, "23_real_fault_outputs.csv"))
        cases = pd.concat([old_c[~old_c.fault.isin(ONLY)], cases], ignore_index=True)
        outs = pd.concat([old_o[~old_o.fault.isin(ONLY)], outs], ignore_index=True)
    cases.to_csv(os.path.join(RES, "23_real_fault_cases.csv"), index=False)
    outs.to_csv(os.path.join(RES, "23_real_fault_outputs.csv"), index=False)
    pd.set_option("display.width", 220)
    g = cases.groupby(["set", "fault"]).agg(aircraft=("aircraft", "nunique"), cases=("test_case", "size"), structural=("structural", "mean"),
                                           propagated=("propagated", "mean"), observable=("observable", "mean"), es_max_median=("es_max", "median"))
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()

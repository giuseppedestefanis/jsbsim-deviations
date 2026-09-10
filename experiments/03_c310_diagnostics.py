"""Diagnostics after the first c310 pipeline run.
(a) Raw effect size of each fault vs matched-seed baselines (oracle validation by perturbation).
(b) Sensitivity of baseline VSD and detection to K, min_samples_leaf, and an elapsed-time input.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import importlib  # noqa: E402
pipe = importlib.import_module("02_c310_pipeline")
from devsig.run import load_run  # noqa: E402
from devsig import signature as sigmod  # noqa: E402
from devsig.signature import Signatures  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402
from devsig.evaluate import prf, to_category, auc  # noqa: E402

RUNS = pipe.RUNS
FAULT_SEEDS = list(range(100, 110))
KEY = ["position/h-sl-ft", "velocities/vc-kts", "attitude/theta-rad", "attitude/phi-rad", "attitude/psi-rad",
       "velocities/q-rad_sec", "velocities/p-rad_sec", "velocities/r-rad_sec", "aero/alpha-deg",
       "accelerations/qdot-rad_sec2", "accelerations/vdot-ft_sec2", "position/lat-geod-deg"]


def resample(df, hz=1.0):
    t = df["simulation/sim-time-sec"].to_numpy()
    idx = np.searchsorted(t, np.arange(t[0], t[-1], 1.0 / hz))
    return df.iloc[np.clip(idx, 0, len(df) - 1)].reset_index(drop=True)


def main():
    # (a) matched-seed baselines
    jobs = [("baseline", s, None) for s in FAULT_SEEDS]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        pool.map(pipe.one_run, jobs)
    print("== (a) raw effect size: RMS(fault - baseline, same seed) / RMS(baseline seed i - baseline seed j)")
    base = {s: resample(load_run(os.path.join(RUNS, f"baseline_{s:04d}.parquet"))) for s in FAULT_SEEDS}
    rows = []
    for fault in ("inertia", "lift", "pitch", "noop"):
        fl = {s: resample(load_run(os.path.join(RUNS, f"{fault}_{s:04d}.parquet"))) for s in FAULT_SEEDS}
        rec = dict(fault=fault, end_s_fault=np.mean([f["simulation/sim-time-sec"].iloc[-1] for f in fl.values()]).round(0),
                   end_s_base=np.mean([b["simulation/sim-time-sec"].iloc[-1] for b in base.values()]).round(0))
        for m in KEY:
            d_f, d_b = [], []
            for s in FAULT_SEEDS:
                n = min(len(fl[s]), len(base[s]))
                d_f.append(np.sqrt(np.mean((fl[s][m].to_numpy()[:n] - base[s][m].to_numpy()[:n]) ** 2)))
            for i, s in enumerate(FAULT_SEEDS[:-1]):
                s2 = FAULT_SEEDS[i + 1]
                n = min(len(base[s]), len(base[s2]))
                d_b.append(np.sqrt(np.mean((base[s][m].to_numpy()[:n] - base[s2][m].to_numpy()[:n]) ** 2)))
            rec[m.split("/")[-1]] = round(float(np.mean(d_f) / (np.mean(d_b) + 1e-12)), 2)
        rows.append(rec)
    eff = pd.DataFrame(rows).set_index("fault")
    print(eff.to_string())
    # also: max alpha reached in baseline (does the pitch fault region alpha>8 deg ever occur?)
    print("baseline max alpha-deg per run:", [round(float(b["aero/alpha-deg"].max()), 1) for b in base.values()])

    # (b) sensitivity
    print("\n== (b) sensitivity: baseline VSD and category-level recall/precision vs initial oracle")
    paths = {v: sorted(p for p in os.listdir(RUNS) if p.startswith(v + "_")) for v in ("baseline", "inertia", "lift", "pitch", "noop")}
    allbase = [load_run(os.path.join(RUNS, p)) for p in paths["baseline"] if int(p[-12:-8]) < 40]
    train, held = allbase[:20], allbase[20:]
    cands = {f: [load_run(os.path.join(RUNS, p)) for p in paths[f]] for f in ("inertia", "lift", "pitch", "noop")}
    out = []
    for time_input in (False, True):
        sigmod.INPUT_PREFIXES = tuple(x for x in sigmod.INPUT_PREFIXES if x != "simulation/") + (("simulation/sim-time-sec",) if time_input else ())
        for k in (3, 5, 7):
            for msl in (20, 200):
                sig = Signatures(k=k, min_samples_leaf=msl).fit(train)
                null = NullModel(pd.DataFrame([sig.vsd(r) for r in held]))
                thr = null.threshold("p95")
                rec = dict(time_input=time_input, k=k, msl=msl, base_vsd=round(float(null.vsd_base.mean()), 3),
                           thr_mean=round(float(thr.mean()), 3), leaves=int(sig.n_rules().median()))
                for f in ("inertia", "lift", "pitch", "noop"):
                    rsd, fl = null.flag_perm(pd.DataFrame([sig.vsd(r) for r in cands[f]]))[:2]
                    if f == "noop":
                        rec["noop_fa"] = round(float(fl.mean()), 2)
                        continue
                    oracle = pd.Series({m: bool(o[f]) for m, o in pipe.ORACLE_C310.items()})
                    idx = [m for m in oracle.index if m in fl.index]
                    c = prf(to_category(fl[idx]), to_category(oracle[idx]))
                    rec[f"{f}_R/P"] = f"{c['recall']:.2f}/{c['precision']:.2f}" if not np.isnan(c["precision"]) else f"{c['recall']:.2f}/-"
                    rec[f"{f}_auc"] = round(auc(rsd[idx], oracle[idx]), 2)
                out.append(rec)
                print(rec, flush=True)
    pd.DataFrame(out).to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "03_c310_sensitivity.csv"), index=False)
    eff.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "03_c310_effect_size.csv"))


if __name__ == "__main__":
    main()

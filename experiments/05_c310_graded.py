"""Graded fault magnitudes on the c310 circuit with an effect-size gate before scoring.

For each fault instance: 10 runs (seeds 100-109, matched to baseline seeds 100-109), raw effect size in
take-off (2-120 s) and cruise (120-1800 s), then signature detection (K=5, whole-run and 30 s windows)
against a physics-derived metric-level oracle.
"""
import importlib
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
pipe = importlib.import_module("02_c310_pipeline")
w4 = importlib.import_module("04_c310_windowed")
from devsig import mutate  # noqa: E402
from devsig.signature import Signatures, category  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402
from devsig.evaluate import prf, to_category, auc  # noqa: E402

TCOL = "simulation/sim-time-sec"
SEEDS = list(range(100, 110))

# fault instances: name -> edit
INSTANCES = {
    "inertia_x1.3": mutate.chain(mutate.scale_tag("iyy", 1.3), mutate.add_ixz(600.0)),
    "inertia_x1.6": mutate.chain(mutate.scale_tag("iyy", 1.6), mutate.add_ixz(1000.0)),
    "inertia_x2.0": mutate.chain(mutate.scale_tag("iyy", 2.0), mutate.add_ixz(1500.0)),
    "lift_+0.05": mutate.offset_table_column("aero/coefficient/CLo", 0.05),
    "lift_+0.20": mutate.offset_table_column("aero/coefficient/CLo", 0.20),
    "clalpha_x0.8": mutate.scale_function_value("aero/coefficient/CLalpha", 0.8),
    "clalpha_x1.2": mutate.scale_function_value("aero/coefficient/CLalpha", 1.2),
    "pitchnl_s0.0": mutate.nonlinear_cmalpha(8.0, 0.0),
    "pitchnl_s-0.5": mutate.nonlinear_cmalpha(8.0, -0.5),
    "cmalpha_x0.7": mutate.scale_function_value("aero/coefficient/Cmalpha", 0.7),
    "cmalpha_x0.4": mutate.scale_function_value("aero/coefficient/Cmalpha", 0.4),
    "cmq_x0.5": mutate.scale_function_value("aero/coefficient/Cmq", 0.5),
}
# already generated in 02: inertia (x1.15, ixz 300), lift (+0.10), pitch (s0.3)
EXISTING = {"inertia": "inertia_x1.15", "lift": "lift_+0.10", "pitch": "pitchnl_s0.3"}

LONG = ["position/h-sl-ft", "attitude/theta-rad", "attitude/pitch-rad", "velocities/q-rad_sec", "accelerations/qdot-rad_sec2",
        "aero/alpha-deg", "velocities/w-fps", "velocities/h-dot-fps", "velocities/u-fps", "velocities/vc-kts", "velocities/vt-fps",
        "accelerations/udot-ft_sec2", "accelerations/wdot-ft_sec2"]
LAT = ["attitude/phi-rad", "attitude/roll-rad", "velocities/p-rad_sec", "velocities/r-rad_sec", "accelerations/pdot-rad_sec2",
       "accelerations/rdot-rad_sec2", "aero/beta-deg", "velocities/v-fps", "accelerations/vdot-ft_sec2", "attitude/psi-rad"]
ORACLE = {  # physics-derived: metrics with a direct or first-order effect
    "inertia": set(LONG) | set(LAT),   # Iyy: pitch dynamics; Ixz: roll-yaw coupling
    "lift": set(LONG),
    "clalpha": set(LONG),
    "pitchnl": set(LONG),
    "cmalpha": set(LONG),
    "cmq": set(LONG),
}


def fam(name):
    return name.split("_")[0]


def effect_size(cand, base, lo, hi, metrics):
    out = {}
    for m in metrics:
        df_, db_ = [], []
        for i in range(len(cand)):
            a, b = cand[i], base[i]
            ma = (a[TCOL] >= lo) & (a[TCOL] < hi); mb = (b[TCOL] >= lo) & (b[TCOL] < hi)
            n = min(ma.sum(), mb.sum())
            df_.append(np.sqrt(np.mean((a.loc[ma, m].to_numpy()[:n] - b.loc[mb, m].to_numpy()[:n]) ** 2)))
            if i + 1 < len(base):
                b2 = base[i + 1]; mb2 = (b2[TCOL] >= lo) & (b2[TCOL] < hi); n2 = min(mb.sum(), mb2.sum())
                db_.append(np.sqrt(np.mean((b.loc[mb, m].to_numpy()[:n2] - b2.loc[mb2, m].to_numpy()[:n2]) ** 2)))
        out[m] = float(np.mean(df_) / (np.mean(db_) + 1e-12))
    return pd.Series(out)


def main():
    apaths = {n: mutate.make_variant("c310", n, pipe.VARIANTS, e) for n, e in INSTANCES.items()}
    jobs = [(n, s, apaths[n]) for n in INSTANCES for s in SEEDS]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        res = pool.map(pipe.one_run, jobs)
    failed = [r for r in res if r[2] is None]
    if failed:
        print("FAILED runs:", failed)
    base = w4.load("baseline", range(40)); train, held = base[:20], base[20:]
    base_m = w4.load("baseline", SEEDS)
    sig = Signatures(k=5).fit(train)
    null = NullModel(pd.DataFrame([sig.vsd(r) for r in held]))
    thr = null.threshold("p95")
    W = 30.0
    tr_w = [sig.vsd_windowed(r, W) for r in train]
    base_w = pd.concat(tr_w).groupby(level=0).mean()
    def rsd_w(r):
        v = sig.vsd_windowed(r, W); return v - base_w.reindex(v.index).fillna(base_w.mean())
    thr_w = pd.DataFrame([rsd_w(r).max(axis=0) for r in held]).quantile(0.95, axis=0)

    rows = []
    names = list(EXISTING.items()) + [(n, n) for n in INSTANCES]
    for stored, label in names:
        cand = w4.load(stored, [s for s in SEEDS if os.path.exists(os.path.join(pipe.RUNS, f"{stored}_{s:04d}.parquet"))])
        if len(cand) < 5:
            print(label, "too few runs"); continue
        f = fam(label); oracle_set = ORACLE[f]
        es_to = effect_size(cand, base_m, 2, 120, sig.outputs); es_cr = effect_size(cand, base_m, 120, 1800, sig.outputs)
        es = pd.concat([es_to, es_cr], axis=1).max(axis=1)
        gate = es > 1.0  # metrics where the fault moves the raw data more than benign variation
        oracle = pd.Series({m: (m in oracle_set) for m in sig.outputs})
        oracle_gated = oracle & gate
        vs = pd.DataFrame([sig.vsd(r) for r in cand])
        rsd, fl = null.flag_perm(vs)[:2]
        rsd_win, fl_win = flag(pd.DataFrame([rsd_w(r).max(axis=0) for r in cand]), thr_w)
        rec = dict(fault=label, n=len(cand), es_takeoff_max=round(float(es_to.max()), 2), es_cruise_max=round(float(es_cr.max()), 2),
                   n_oracle=int(oracle.sum()), n_oracle_gated=int(oracle_gated.sum()),
                   n_flag_whole=int(fl.sum()), n_flag_win=int(fl_win.sum()))
        for tag, flg, score in (("whole", fl, rsd), ("win30", fl_win, rsd_win)):
            for otag, orc in (("phys", oracle), ("gated", oracle_gated)):
                if orc.sum() == 0:
                    rec[f"{tag}_{otag}_R/P"] = "-"
                    continue
                m = prf(flg, orc); c = prf(to_category(flg), to_category(orc))
                rec[f"{tag}_{otag}_R/P"] = f"{m['recall']:.2f}/{(m['precision'] if not np.isnan(m['precision']) else 0):.2f}"
                rec[f"{tag}_{otag}_catR/P"] = f"{c['recall']:.2f}/{(c['precision'] if not np.isnan(c['precision']) else 0):.2f}"
                rec[f"{tag}_{otag}_auc"] = round(auc(score, orc), 2)
        rec["flagged_win"] = ",".join(m.split("/")[-1] for m in fl_win[fl_win].index)
        rows.append(rec)
        print(rec, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "05_c310_graded.csv"), index=False)
    print("\n" + out[["fault", "es_takeoff_max", "es_cruise_max", "n_oracle_gated", "n_flag_whole", "n_flag_win",
                       "whole_phys_R/P", "win30_phys_R/P", "win30_gated_R/P", "win30_gated_catR/P", "win30_gated_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()

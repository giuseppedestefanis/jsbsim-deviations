"""Tier A battery on selected aircraft with dynamic faults: do open-loop manoeuvres expose what the circuit hid?"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from devsig.run import load_run  # noqa: E402
from devsig import mutate  # noqa: E402
from devsig.manoeuvres import BATTERY, load_tier_a_config, tier_a_run  # noqa: E402
from devsig.signature import Signatures  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402
from devsig.evaluate import prf, to_category, auc  # noqa: E402

AIRCRAFT = sys.argv[1:] or ["c310", "737"]
RUNS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "runs", "tier_a")
VARIANTS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "variants")
N_BASE, N_FAULT = 40, 10
FAULTS = {  # name -> edit factory (axis-level operators work across naming conventions; skipped if no match)
    "inertia_x1.6": lambda: mutate.chain(mutate.scale_tag("iyy", 1.6), mutate.add_ixz(1000.0)),
    "cmq_x0.5": lambda: mutate.scale_axis_function("PITCH", mutate.PITCH_DAMP, 0.5),
    "cmalpha_x0.7": lambda: mutate.scale_axis_function("PITCH", mutate.PITCH_STIFF, 0.7),
    "clo_+0.1": lambda: mutate.add_axis_term("LIFT", "aero/force/Lift_offset_injected", 0.1),
    "noop": lambda: mutate.noop_rewrite(),
}
LONG = ["position/h-sl-ft", "position/h-agl-ft", "attitude/theta-rad", "velocities/q-rad_sec", "accelerations/qdot-rad_sec2",
        "aero/alpha-deg", "velocities/w-fps", "velocities/h-dot-fps", "velocities/u-fps", "velocities/vc-kts", "velocities/vt-fps",
        "accelerations/udot-ft_sec2", "accelerations/wdot-ft_sec2"]
LAT = ["attitude/phi-rad", "velocities/p-rad_sec", "velocities/r-rad_sec", "accelerations/pdot-rad_sec2",
       "accelerations/rdot-rad_sec2", "aero/beta-deg", "velocities/v-fps", "accelerations/vdot-ft_sec2", "attitude/psi-rad"]
ORACLE = {"inertia": set(LONG) | set(LAT), "cmq": set(LONG), "cmalpha": set(LONG), "clo": set(LONG)}


def one(args):
    aircraft, cfg, variant, manoeuvre, seed, apath = args
    path = os.path.join(RUNS, aircraft, f"{variant}_{manoeuvre}_{seed:04d}.parquet")
    if os.path.exists(path):
        return aircraft, variant, manoeuvre, seed, path, "cached"
    df, msg, var = tier_a_run(aircraft, cfg, manoeuvre, seed, root_dir=apath)
    if df is None:
        return aircraft, variant, manoeuvre, seed, None, msg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    return aircraft, variant, manoeuvre, seed, path, "ok"


def effect_size(cand, base, metrics):
    out = {}
    for m in metrics:
        df_ = [np.sqrt(np.mean((cand[i][m].to_numpy()[:min(len(cand[i]), len(base[i]))] - base[i][m].to_numpy()[:min(len(cand[i]), len(base[i]))]) ** 2)) for i in range(min(len(cand), len(base)))]
        db_ = [np.sqrt(np.mean((base[i][m].to_numpy()[:min(len(base[i]), len(base[i + 1]))] - base[i + 1][m].to_numpy()[:min(len(base[i]), len(base[i + 1]))]) ** 2)) for i in range(len(base) - 1)]
        # floor on the benign spread: 0.1% of the output's pooled standard deviation over the baseline runs,
        # so the ratio stays defined for outputs whose benign variation is near zero
        floor = 1e-3 * float(np.concatenate([b[m].to_numpy() for b in base]).std()) + 1e-12
        out[m] = float(np.mean(df_) / max(np.mean(db_), floor))
    return pd.Series(out)


def main():
    cfgs = load_tier_a_config()["tier_a"]
    rows = []
    for ac in AIRCRAFT:
        cfg = cfgs[ac]
        apaths, skipped = {}, []
        for name, factory in FAULTS.items():
            try:
                apaths[name] = mutate.make_variant(ac, f"{ac}_{name}", VARIANTS, factory())
            except RuntimeError as e:
                skipped.append(f"{name}: {e}")
        if skipped:
            print(ac, "skipped faults:", skipped)
        jobs = []
        for man in BATTERY:
            jobs += [(ac, cfg, "baseline", man, s, None) for s in range(N_BASE)]
            jobs += [(ac, cfg, "baseline", man, 100 + s, None) for s in range(N_FAULT)]  # matched seeds for effect size
            for name in apaths:
                jobs += [(ac, cfg, name, man, 100 + s, apaths[name]) for s in range(N_FAULT)]
        with Pool(max(1, os.cpu_count() - 1)) as pool:
            res = pool.map(one, jobs, chunksize=4)
        fails = [r for r in res if r[4] is None]
        if fails:
            msgs = pd.Series([r[5] for r in fails]).value_counts()
            print(ac, f"{len(fails)} failed runs:", msgs.to_dict())
        paths = {}
        for _, variant, man, seed, path, _ in res:
            if path:
                paths.setdefault((variant, man), {})[seed] = path
        for man in BATTERY:
            bp = paths.get(("baseline", man), {})
            base = [load_run(bp[s]) for s in range(N_BASE) if s in bp]
            base_m = [load_run(bp[s]) for s in range(100, 110) if s in bp]
            if len(base) < 30 or len(base_m) < 8:
                print(ac, man, "too few baseline runs", len(base), len(base_m)); continue
            train, held = base[: len(base) // 2], base[len(base) // 2:]
            sig = Signatures(k=5, min_samples_leaf=10).fit(train)
            null = NullModel(pd.DataFrame([sig.vsd(r) for r in held]))
            thr = null.threshold("p95")
            for name in list(apaths):
                cp = paths.get((name, man), {})
                cand = [load_run(cp[s]) for s in range(100, 110) if s in cp]
                if len(cand) < 8:
                    continue
                rsd, fl = null.flag_perm(pd.DataFrame([sig.vsd(r) for r in cand]))[:2]
                rec = dict(aircraft=ac, manoeuvre=man, fault=name, n_flag=int(fl.sum()), base_vsd=round(float(null.vsd_base.mean()), 3))
                if name == "noop":
                    rec["false_alarm"] = round(float(fl.mean()), 2)
                else:
                    es = effect_size(cand, base_m, sig.outputs)
                    orc = pd.Series({m: (m in ORACLE[name.split("_")[0]]) for m in sig.outputs})
                    gated = orc & (es > 1.0)
                    rec["es_max"] = round(float(es.max()), 2); rec["n_gated"] = int(gated.sum())
                    for tag, o in (("phys", orc), ("gated", gated)):
                        if o.sum() == 0:
                            rec[f"{tag}_R/P"] = "-"; continue
                        m = prf(fl, o); c = prf(to_category(fl), to_category(o))
                        rec[f"{tag}_R/P"] = f"{m['recall']:.2f}/{(0 if np.isnan(m['precision']) else m['precision']):.2f}"
                        rec[f"{tag}_cat"] = f"{c['recall']:.2f}/{(0 if np.isnan(c['precision']) else c['precision']):.2f}"
                        rec[f"{tag}_auc"] = round(auc(rsd, o), 2)
                rows.append(rec)
        out = pd.DataFrame([r for r in rows if r["aircraft"] == ac])
        print(f"\n== {ac}")
        print(out.to_string(index=False))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), f"06_tier_a_battery_{'_'.join(AIRCRAFT)}.csv"), index=False)


if __name__ == "__main__":
    main()

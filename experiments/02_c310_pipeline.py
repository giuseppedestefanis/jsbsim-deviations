"""Phase 1: end-to-end signature pipeline on the c310 round trip with the three initial faults.

Generates baseline runs with benign variation, faulty runs for inertia / lift / pitch (+ a no-op variant),
trains signatures, derives null thresholds, flags metrics, and scores against the initial oracle.
"""
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from devsig.run import load_run  # noqa: E402
from devsig import mutate, variation  # noqa: E402
from devsig.evaluate import auc, prf, to_category  # noqa: E402
from devsig.run import run_script  # noqa: E402
from devsig.signature import Signatures, category  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402

RUNS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "runs", "c310")
VARIANTS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "variants")
SCRIPT = "scripts/c3104.xml"
SAMPLE_HZ = 5.0
N_BASE, N_FAULT = 40, 10

FAULTS = {
    "inertia": mutate.chain(mutate.scale_tag("iyy", 1.15), mutate.add_ixz(300.0)),
    "lift": mutate.offset_table_column("aero/coefficient/CLo", 0.10),
    "pitch": mutate.nonlinear_cmalpha(alpha_star_deg=8.0, slope_factor_beyond=0.3),
    "noop": mutate.noop_rewrite(),
}

# Oracle for the three initial faults (X = deviation) per output metric, mapped to JSBSim property names.
ORACLE_C310 = {
    "position/h-sl-ft":              dict(inertia=1, lift=0, pitch=1),
    "position/lat-geod-deg":         dict(inertia=0, lift=1, pitch=0),
    "attitude/pitch-rad":            dict(inertia=0, lift=0, pitch=1),
    "attitude/roll-rad":             dict(inertia=0, lift=1, pitch=0),
    "attitude/theta-rad":            dict(inertia=0, lift=0, pitch=1),
    "velocities/u-fps":              dict(inertia=1, lift=1, pitch=1),
    "velocities/v-fps":              dict(inertia=1, lift=1, pitch=1),
    "velocities/w-fps":              dict(inertia=1, lift=0, pitch=0),
    "velocities/p-rad_sec":          dict(inertia=1, lift=1, pitch=0),
    "velocities/q-rad_sec":          dict(inertia=0, lift=0, pitch=1),
    "velocities/r-rad_sec":          dict(inertia=0, lift=0, pitch=0),
    "accelerations/pdot-rad_sec2":   dict(inertia=1, lift=1, pitch=0),
    "accelerations/qdot-rad_sec2":   dict(inertia=0, lift=0, pitch=1),
    "accelerations/rdot-rad_sec2":   dict(inertia=0, lift=0, pitch=0),
    "accelerations/vdot-ft_sec2":    dict(inertia=1, lift=1, pitch=1),
}


def one_run(args):
    variant, seed, aircraft_path = args
    path = os.path.join(RUNS, f"{variant}_{seed:04d}.parquet")
    if os.path.exists(path):
        return variant, seed, path, "cached"
    v = variation.draw(seed)
    r = run_script(SCRIPT, sample_hz=SAMPLE_HZ, max_wall_s=120, root_dir=aircraft_path,
                   post_ic=lambda fdm: variation.apply(fdm, v))
    if not r.ok or len(r.df) < 100:
        return variant, seed, None, f"FAILED {r.error} rows={len(r.df)}"
    r.df.attrs["variation"] = v.as_dict()
    r.df.to_parquet(path)
    return variant, seed, path, f"ok end={r.sim_end_s:.0f}s"


def load(paths):
    return [load_run(p) for p in paths]


def main():
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(VARIANTS, exist_ok=True)
    apaths = {"baseline": None}
    for name, edit in FAULTS.items():
        apaths[name] = mutate.make_variant("c310", name, VARIANTS, edit)
    jobs = [("baseline", s, None) for s in range(N_BASE)]
    for name in FAULTS:
        jobs += [(name, 100 + s, apaths[name]) for s in range(N_FAULT)]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        results = pool.map(one_run, jobs)
    paths = {}
    for variant, seed, path, msg in results:
        if path:
            paths.setdefault(variant, []).append(path)
        if "FAILED" in msg:
            print(f"  {variant} seed {seed}: {msg}")
    print({k: len(v) for k, v in paths.items()})

    base = load(sorted(paths["baseline"]))
    train, held = base[: N_BASE // 2], base[N_BASE // 2:]
    sig = Signatures(k=3).fit(train)
    print(f"signatures: {len(sig.inputs)} inputs, {len(sig.outputs)} outputs; leaves per tree median={sig.n_rules().median():.0f}")
    null = NullModel(pd.DataFrame([sig.vsd(r) for r in held]))
    print("baseline VSD (mean over held-out):"); print(null.vsd_base.round(3).to_string())

    rows = []
    for name in FAULTS:
        cand = pd.DataFrame([sig.vsd(r) for r in load(sorted(paths[name]))])
        rsd_runs = null.rsd(cand)
        for thr_name in ("p95", "max"):
            thr = null.threshold(thr_name)
            rsd, flagged = flag(rsd_runs, thr)
            if name == "noop":
                fa = flagged.mean()
                rows.append(dict(fault=name, threshold=thr_name, level="metric", false_alarm_rate=round(float(fa), 3), n_flagged=int(flagged.sum())))
                continue
            oracle = pd.Series({m: bool(o[name]) for m, o in ORACLE_C310.items()})
            m_idx = [m for m in oracle.index if m in flagged.index]
            res_m = prf(flagged[m_idx], oracle[m_idx])
            res_c = prf(to_category(flagged[m_idx]), to_category(oracle[m_idx]))
            rows.append(dict(fault=name, threshold=thr_name, level="metric", **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in res_m.items()},
                             auc=round(auc(rsd[m_idx], oracle[m_idx]), 2), flagged=",".join(m.split("/")[-1] for m in flagged[m_idx][flagged[m_idx]].index)))
            rows.append(dict(fault=name, threshold=thr_name, level="category", **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in res_c.items()}))
        if name != "noop":
            print(f"\n== {name}: median RSD per metric (oracle X marked)")
            show = pd.DataFrame({"rsd": rsd.round(3), "thr_p95": null.threshold("p95").round(3), "thr_max": null.threshold("max").round(3)})
            show["oracle"] = ["X" if ORACLE_C310.get(m, {}).get(name, 0) else "" for m in show.index]
            print(show[show.index.isin(ORACLE_C310)].to_string())
    out = pd.DataFrame(rows)
    print("\n== Summary (metric and category level)")
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "02_c310_pipeline.csv"), index=False)


if __name__ == "__main__":
    main()

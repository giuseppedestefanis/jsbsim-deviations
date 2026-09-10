"""Tier B run generation: closed-loop missions with phases on the autopilot aircraft, plus the AH-1S stock flight test.
Runs: 40 baseline (seeds 0-39) + 10 matched baseline (100-109) + 10 per fault variant (100-109).
Files: results/runs/tier_b/<ac>/<variant>_<mission>_<seed>.parquet (with a 'phase' column)."""
import os
import json
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from devsig import mutate, variation  # noqa: E402
from devsig.missions import MISSIONS, tier_b_run  # noqa: E402
from devsig.run import run_script  # noqa: E402

RUNS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "runs", "tier_b")
VARIANTS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "variants")
CFG = json.load(open(os.path.join(HERE, "..", "config", "tier_b_aircraft.json")))
FAULTS = {
    "inertia_x1.6": lambda: mutate.chain(mutate.scale_tag("iyy", 1.6), mutate.add_ixz(1000.0)),
    "cmq_x0.5": lambda: mutate.scale_axis_function("PITCH", mutate.PITCH_DAMP, 0.5),
    "cmalpha_x0.7": lambda: mutate.scale_axis_function("PITCH", mutate.PITCH_STIFF, 0.7),
    "clo_+0.1": lambda: mutate.add_axis_term("LIFT", "aero/force/Lift_offset_injected", 0.1),
    "B_elevator_drag_8819410c": mutate.revert_elevator_drag_fix,
    "noop": lambda: mutate.noop_rewrite(),
}
AH1S_FAULTS = {"inertia_x1.6": FAULTS["inertia_x1.6"], "noop": FAULTS["noop"]}


def one(args):
    kind, ac, cfg, variant, mission, seed, root = args
    path = os.path.join(RUNS, ac, f"{variant}_{mission}_{seed:04d}.parquet")
    if os.path.exists(path):
        return ac, variant, mission, "cached"
    if kind == "mission":
        df, msg, _ = tier_b_run(ac, cfg, mission, seed, root_dir=root)
        if df is None:
            return ac, variant, mission, msg
    else:  # ah1s script with benign variation and time phases
        v = variation.draw(seed)
        r = run_script(cfg["script"], sample_hz=10.0, max_wall_s=300, root_dir=root, post_ic=lambda fdm: variation.apply(fdm, v))
        if not r.ok or len(r.df) < 1000:
            return ac, variant, mission, f"FAILED {r.error} rows={len(r.df)}"
        df = r.df.rename(columns={"simulation/sim-time-sec": "t"}) if "t" not in r.df else r.df
        t = df["t"] if "t" in df else df["simulation/sim-time-sec"]
        bounds = CFG["ah1s_phases"]
        labels = pd.cut(t, bins=[b[0] for b in bounds], labels=[b[1] for b in bounds[:-1]], right=False, include_lowest=True)
        df.insert(1, "phase", labels.astype(str))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    return ac, variant, mission, "ok"


def main():
    only = sys.argv[1:]  # optional aircraft subset
    jobs = []
    for ac, c in CFG["tier_b"].items():
        if only and ac not in only:
            continue
        base_root = os.path.abspath(c["root"]) if c.get("root") else None
        for mission in CFG["missions"]:
            jobs += [("mission", ac, c, "baseline", mission, s, base_root) for s in list(range(40)) + list(range(100, 110))]
            for name, factory in FAULTS.items():
                try:
                    root = mutate.make_variant(ac, f"tb_{ac}_{name}", VARIANTS, factory(), src_root=base_root)
                except RuntimeError as e:
                    print(ac, name, "skipped:", e); continue
                jobs += [("mission", ac, c, name, mission, s, root) for s in range(100, 110)]
    if not only or "ah1s" in only:
        c = CFG["ah1s"]
        jobs += [("script", "ah1s", c, "baseline", "flight_test", s, None) for s in list(range(40)) + list(range(100, 110))]
        for name, factory in AH1S_FAULTS.items():
            root = mutate.make_variant("ah1s", f"tb_ah1s_{name}", VARIANTS, factory())
            jobs += [("script", "ah1s", c, name, "flight_test", s, root) for s in range(100, 110)]
    print(len(jobs), "runs to do", flush=True)
    with Pool(int(os.environ.get("WORKERS", max(1, os.cpu_count() - 2)))) as pool:
        res = pool.map(one, jobs, chunksize=2)
    import collections
    print(collections.Counter((a, v, m) for a, v, m, s in res if s not in ("ok", "cached")))
    print("ok:", sum(1 for r in res if r[3] == "ok"), "cached:", sum(1 for r in res if r[3] == "cached"), "of", len(res))


if __name__ == "__main__":
    main()

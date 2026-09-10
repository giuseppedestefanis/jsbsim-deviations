"""Coverage-driven follow-up test for the ground-effect fault (Family B, 9c058118, Boeing 737): a low pass.
The 737 is trimmed at 40 ft above ground with half flaps at 160 knots and flown for 30 s with an elevator pulse,
which takes it through the height band (h/b about 0.4) where the reverted lift table differs. Designed after the
Tier A battery missed the fault, so it is reported as a coverage test, not as part of the battery.
Runs: 40 baseline, 10 matched baseline (seeds 100-109), 10 no-op, 10 ground-effect. Scored like Tier A:
regression residual and envelope with pseudo-candidate thresholds, physics oracle (longitudinal), effect-size gate."""
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
from devsig import variation  # noqa: E402
from devsig.tier_a import fly, load_and_trim  # noqa: E402
from devsig.manoeuvres import enable_turbulence, salt  # noqa: E402
from devsig.run import DEFAULT_PROPS  # noqa: E402
from devsig.baselines import Envelope, RegressionResidual  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402
from devsig.evaluate import prf, auc  # noqa: E402
b6 = __import__("06_tier_a_battery")

RUNS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "runs", "coverage", "737")
VARIANTS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "variants")
H_AGL, VC, FLAP = 40.0, 160.0, 0.5


def one(args):
    variant, seed = args[:2]
    root_override = args[2] if len(args) > 2 else None
    path = os.path.join(RUNS, f"{variant}_low_pass_{seed:04d}.parquet")
    if os.path.exists(path):
        return variant, seed, "cached"
    root = root_override or (None if variant == "baseline" else os.path.abspath(os.path.join(VARIANTS, f"737_{variant}")))
    s_var, s_ic = np.random.SeedSequence([seed] + salt("737", "low_pass")).spawn(2)
    rng = np.random.default_rng(s_ic)
    v = variation.draw(seed, rng=np.random.default_rng(s_var))
    h = H_AGL + rng.uniform(-5, 5); vc = VC + rng.uniform(-5, 5)

    def post_ic(fdm):
        variation.apply(fdm, v)
        fdm["fcs/flap-cmd-norm"] = FLAP
    fdm, ok, msg = load_and_trim("737", h, vc, root_dir=root, post_ic=post_ic)
    if not ok:
        return variant, seed, "trim failed: " + msg
    enable_turbulence(fdm, seed)
    e0 = fdm["fcs/elevator-cmd-norm"]
    t1 = 5.0 + rng.uniform(-0.3, 0.3); amp = -0.05 * rng.uniform(0.9, 1.1)
    df = fly(fdm, 30.0, sample_hz=10.0, inputs=[(t1, "fcs/elevator-cmd-norm", e0 + amp), (t1 + 2.0, "fcs/elevator-cmd-norm", e0)], props=DEFAULT_PROPS)
    if df.isna().any().any() or (df["position/h-agl-ft"] <= 0).any():
        return variant, seed, "ground contact"
    os.makedirs(RUNS, exist_ok=True); df.to_parquet(path)
    return variant, seed, "ok"


def load(variant, seeds):
    out = []
    for s in seeds:
        p = os.path.join(RUNS, f"{variant}_low_pass_{s:04d}.parquet")
        if os.path.exists(p):
            out.append(load_run(p))
    return out


def main():
    jobs = [("baseline", s) for s in list(range(40)) + list(range(100, 110))] + [(v, s) for v in ("noop", "B_ground_effect_9c058118") for s in range(100, 110)]
    with Pool(max(1, min(8, os.cpu_count() - 1))) as pool:
        for v, s, msg in pool.imap_unordered(one, jobs):
            if msg != "ok" and msg != "cached":
                print(v, s, msg, flush=True)
    base = load("baseline", range(40)); base_m = load("baseline", range(100, 110))
    train, held = base[:20], base[20:]
    rows = []
    for name, model in (("regression", RegressionResidual(max_iter=60).fit(train)), ("envelope", Envelope(0.05).fit(train))):
        null = NullModel(pd.DataFrame([model.score(r) for r in held])); thr = null.threshold("p95")
        for v in ("noop", "B_ground_effect_9c058118"):
            cand = load(v, range(100, 110))
            rsd, fl = null.flag_perm(pd.DataFrame([model.score(r) for r in cand]))[:2]
            rec = dict(method=name, variant=v, n_flagged=int(fl.sum()))
            if v == "noop":
                rec["false_alarm"] = float(fl.mean())
            else:
                orc = pd.Series({m: (m in b6.LONG) for m in fl.index})
                es = b6.effect_size(cand, base_m, list(fl.index)); gated = orc & (es > 1.0)
                m = prf(fl, orc); rec.update(recall=m["recall"], precision=m["precision"], auc=auc(rsd, orc), es_max=float(es.max()), n_gated=int(gated.sum()))
                if gated.sum():
                    g = prf(fl, gated); rec.update(gated_recall=g["recall"], gated_auc=auc(rsd, gated))
                rec["flagged"] = ";".join(fl[fl].index)
            rows.append(rec)
    out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "20_coverage_low_pass.csv"), index=False)
    pd.set_option("display.width", 200); print(out.to_string())


def run_variant(variant, root=None):
    """Generate the ten candidate runs (seeds 100-109) of one fault variant with the current JSBSim library (set
    PYTHONPATH for a reverted build) and an optional model root (a mirror root for model-file faults)."""
    with Pool(max(1, min(8, os.cpu_count() - 1))) as pool:
        for v, s, msg in pool.imap_unordered(one, [(variant, s, root) for s in range(100, 110)]):
            if msg not in ("ok", "cached"):
                print(v, s, msg, flush=True)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "run":
    run_variant(sys.argv[2], os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None)
    sys.exit(0)

if __name__ == "__main__":
    main()

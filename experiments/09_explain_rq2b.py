"""RQ2b on cached Tier A runs: detector (regression residual) flags; dynamic vs static signatures explain.
Attribution correctness (attributed raw input is the manoeuvre's driving input) and agreement."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import importlib
import os
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
from devsig.explain import attribute, correctness, agreement, agreement_null_median, driving_inputs, raw_input  # noqa: E402


def largest_movement_input(cand):
    """Attribution baseline: the command input with the largest range within the candidate runs."""
    cmds = [c for c in cand[0].columns if c.startswith("fcs/") and c.endswith("cmd-norm") or c in ("propulsion/cutoff_cmd", "propulsion/magneto_cmd")]
    rng = {c: float(np.mean([r[c].max() - r[c].min() for r in cand])) for c in cmds}
    return max(rng, key=rng.get) if rng else None


def changing_inputs(cand):
    cmds = [c for c in cand[0].columns if c.startswith("fcs/") and c.endswith("cmd-norm") or c in ("propulsion/cutoff_cmd", "propulsion/magneto_cmd")]
    return [c for c in cmds if np.mean([r[c].max() - r[c].min() for r in cand]) > 1e-6]


def residual_importance_input(det, cand, output, rng):
    """Attribution from the detector itself: the raw input whose (permuted) features raise the residual most."""
    d = pd.concat([det._prep(r) for r in cand], ignore_index=True)
    X = d[det.features].to_numpy(dtype=float); y = d[output].to_numpy(dtype=float)
    m = det.models[output]
    base = float(np.sqrt(np.mean((m.predict(X) - y) ** 2)))
    best, best_gain = None, -1.0
    for raw in {raw_input(f) for f in det.features}:
        cols = [i for i, f in enumerate(det.features) if raw_input(f) == raw]
        Xp = X.copy(); Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
        gain = float(np.sqrt(np.mean((m.predict(Xp) - y) ** 2))) - base
        if gain > best_gain:
            best, best_gain = raw, gain
    return best

FAULTS = ("inertia_x1.6", "cmq_x0.5", "cmalpha_x0.7", "clo_+0.1")


def one_pair(args):
    ac, man = args
    rows = []
    base = b7.load(ac, "baseline", man, range(40))
    if len(base) < 30:
        return rows
    train, held = base[:len(base) // 2], base[len(base) // 2:]
    det = RegressionResidual(max_iter=60).fit(train)
    dnull = NullModel(pd.DataFrame([det.score(r) for r in held])); dthr = dnull.threshold("p95")
    sigs = {"dynamic": Signatures(k=5, min_samples_leaf=10, lags_s=(0.5, 1.0, 2.0, 5.0), since_change=True).fit(train),
            "static": Signatures(k=5, min_samples_leaf=10).fit(train)}
    snull = {k: NullModel(pd.DataFrame([s.vsd(r) for r in held])) for k, s in sigs.items()}
    for f in FAULTS:
        cand = b7.load(ac, f, man, range(100, 110))
        if len(cand) < 8:
            continue
        _, dflags = dnull.flag_perm(pd.DataFrame([det.score(r) for r in cand]))[:2]
        flagged = list(dflags[dflags].index)
        exp_any = driving_inputs(man)
        # attribution baselines, per test case
        lm = largest_movement_input(cand); ch = changing_inputs(cand)
        only = ch[0] if len(ch) == 1 else None
        rng = np.random.default_rng(0)
        cand_det = [det._prep(r) for r in cand]  # history features computed once per pair
        for k, s in sigs.items():
            cand_p = [s._prep(r) for r in cand]; held_p = [s._prep(r) for r in held]
            rsd, sflags = snull[k].flag_perm(pd.DataFrame([s.vsd(r) for r in cand]))[:2]
            agr = agreement(dflags, sflags)
            agr_med = agreement_null_median(dflags, rsd, snull[k].rsd_null.median(axis=0))
            corr, corr_lm, corr_only, corr_imp, stab, n_trim = [], [], [], [], [], 0
            for o in flagged:
                a = attribute(s, cand_p, o, held_p)
                c = correctness(a, man, o)
                if c is not None:
                    corr.append(c)
                    exp = driving_inputs(man, o)
                    corr_lm.append(lm in exp if lm else False)
                    corr_only.append((only in exp) if only else (lm in exp if lm else False))
                    if k == "dynamic":
                        corr_imp.append(residual_importance_input(det, cand_det, o, rng) in exp)
                        per_run = [attribute(s, [r], o, held_p).get("raw") for r in cand_p]
                        stab.append(float(np.mean([p == a.get("raw") for p in per_run])))
                if a.get("trim_shift"):
                    n_trim += 1
            rows.append(dict(aircraft=ac, manoeuvre=man, fault=f, explainer=k, n_flagged=len(flagged),
                             agreement=agr, agreement_null_median=agr_med, n_explained=len(corr), correctness=(np.mean(corr) if corr else np.nan),
                             correctness_largest_movement=(np.mean(corr_lm) if corr_lm else np.nan),
                             correctness_only_changing=(np.mean(corr_only) if corr_only else np.nan),
                             correctness_residual_importance=(np.mean(corr_imp) if corr_imp else np.nan),
                             stability=(np.mean(stab) if stab else np.nan), n_with_trim_shift=n_trim))
    print(ac, man, "done", flush=True)
    return rows


def main():
    aircraft = sys.argv[1:] or ["c310", "737"]
    pairs = [(ac, man) for ac in aircraft for man in BATTERY if man != "cruise"]
    with Pool(max(1, min(8, os.cpu_count() - 1))) as pool:
        results = pool.map(one_pair, pairs, chunksize=1)
    rows = [r for rs in results for r in rs]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "09_explain_rq2b.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n== per aircraft and explainer: mean agreement, mean attribution correctness, flagged metrics per (manoeuvre,fault)")
    print(out.groupby(["aircraft", "explainer"]).agg(agreement=("agreement", "mean"), correctness=("correctness", "mean"),
                                                     n_flagged=("n_flagged", "mean"), n_explained=("n_explained", "sum")).round(2).to_string())
    print("\n== per manoeuvre (dynamic):")
    d = out[out.explainer == "dynamic"]
    print(d.groupby(["aircraft", "manoeuvre"]).agg(agreement=("agreement", "mean"), correctness=("correctness", "mean"), n_flagged=("n_flagged", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()

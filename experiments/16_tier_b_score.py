"""Tier B scoring (RQ4 and the Tier B part of RQ1/RQ2b): closed-loop missions with phases.
Per (aircraft, mission): detector (regression residual) and dynamic/static signatures trained on 20 baseline runs,
null from 20 held-out; variants scored whole-run, per sliding window (30 s, 120 s: max over windows of the
per-window score relative to the training mean), and per phase (score within each phase). Effect-size gate vs
matched-seed baselines. Explainer agreement and correctness with a driving input per phase."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
b6 = importlib.import_module("06_tier_a_battery")
from devsig.run import load_run  # noqa: E402
from devsig.signature import Signatures, split_metrics  # noqa: E402
from devsig.baselines import RegressionResidual, Envelope, KS  # noqa: E402
from devsig.threshold import NullModel, flag  # noqa: E402
from devsig.evaluate import prf, to_category, auc  # noqa: E402
from devsig.explain import attribute, agreement  # noqa: E402

RUNS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "runs", "tier_b")
CFG = json.load(open(os.path.join(HERE, "..", "config", "tier_b_aircraft.json")))
TCOL = "simulation/sim-time-sec"
ORACLE = dict(b6.ORACLE)
ORACLE["B"] = set(b6.LONG)  # elevator drag
# driving inputs per phase (closed loop: the autopilot moves these to follow the setpoint change)
ELEV = {"fcs/elevator-cmd-norm", "fcs/elevator-pos-rad", "fcs/elevator-pos-norm", "fcs/pitch-trim-cmd-norm"}
AIL = {"fcs/aileron-cmd-norm", "fcs/left-aileron-pos-rad", "fcs/right-aileron-pos-rad", "fcs/left-aileron-pos-norm"}
THR = {"fcs/throttle-cmd-norm", "fcs/throttle-cmd-norm[1]"}
DRIVING = {"climb": ELEV | THR, "turn": AIL, "cruise": set(), "slow": THR | ELEV, "descend": ELEV | THR, "turnback": AIL, "level": set(), "settle": set(),
           "config": {"fcs/flap-cmd-norm", "fcs/flap-pos-deg", "fcs/flap-pos-norm"} | THR, "gear": {"gear/gear-cmd-norm", "gear/gear-pos-norm"}, "final": ELEV | THR,
           # ah1s
           "liftoff": set(), "hover": set(), "accelerate": set(), "climb_8000": set(), "fast_cruise": set(), "descend_5000": set(), "end": set()}
WINDOWS = (30.0, 120.0)
# Tier B aircraft whose files JSBSim commit 8819410c changed; the other models always carried the corrected term
ELEV_DRAG_CHANGED = {"737"}


def load(ac, variant, mission, seeds):
    out = []
    for s in seeds:
        p = os.path.join(RUNS, ac, f"{variant}_{mission}_{s:04d}.parquet")
        if os.path.exists(p):
            d = load_run(p)
            if TCOL not in d and "t" in d:
                d[TCOL] = d["t"]
            out.append(d)
    return out


def variants_present(ac, mission):
    d = os.path.join(RUNS, ac)
    if not os.path.isdir(d):
        return []
    names = set()
    for f in os.listdir(d):
        if f"_{mission}_" in f:
            names.add(f.split(f"_{mission}_")[0])
    names.discard("baseline")
    if ac not in ELEV_DRAG_CHANGED:  # score the real fault only where its fix changed the model file
        names.discard("B_elevator_drag_8819410c")
    return sorted(names)


def fam(v):
    return "B" if v.startswith("B_") else v.split("_")[0]


def phase_scores(model, run, kind):
    """Score per phase: dict phase -> Series over outputs."""
    out = {}
    for ph, part in run.groupby("phase", sort=False):
        if len(part) < 20:
            continue
        out[ph] = model.score(part) if kind == "det" else model.vsd(part)
    return out


def windowed_max(model, run, W, kind, base_w):
    v = model.score_windowed(run, W) if kind == "det" else model.vsd_windowed(run, W)
    return (v - base_w.reindex(v.index).fillna(base_w.mean())).max(axis=0)


def one_pair(args):
    ac, mission = args
    rows = []
    base = load(ac, "baseline", mission, range(40)); base_m = load(ac, "baseline", mission, range(100, 110))
    if len(base) < 30 or len(base_m) < 8:
        print(ac, mission, "too few baselines", len(base), len(base_m), flush=True); return rows
    train, held = base[:len(base) // 2], base[len(base) // 2:]
    _, outputs = split_metrics(train[0].columns)
    det = RegressionResidual(max_iter=60).fit(train)
    sigs = {"dynamic": Signatures(k=5, min_samples_leaf=10, lags_s=(0.5, 1.0, 2.0, 5.0), since_change=True).fit(train),
            "static": Signatures(k=5, min_samples_leaf=10).fit(train)}
    comps = {"envelope": Envelope(0.05).fit(train), "ks": KS().fit(train)}
    # whole-run nulls
    nd = NullModel(pd.DataFrame([det.score(r) for r in held])); thd = nd.threshold("p95")
    ns = {k: NullModel(pd.DataFrame([s.vsd(r) for r in held])) for k, s in sigs.items()}
    nc = {k: NullModel(pd.DataFrame([m.score(r) for r in held])) for k, m in comps.items()}
    # windowed nulls (detector and dynamic signature)
    win = {}
    for W in WINDOWS:
        for kind, model in (("det", det), ("sig", sigs["dynamic"])):
            tr = [model.score_windowed(r, W) if kind == "det" else model.vsd_windowed(r, W) for r in train]
            base_w = pd.concat(tr).groupby(level=0).mean()
            nw = NullModel(pd.DataFrame([windowed_max(model, r, W, kind, base_w) for r in held]))
            win[(kind, W)] = (model, base_w, nw)
    # phase nulls
    phases = [p for p in train[0]["phase"].unique()]
    ph_null = {}
    for kind, model in (("det", det), ("sig", sigs["dynamic"])):
        per = [phase_scores(model, r, kind) for r in held]
        for ph in phases:
            vals = [p[ph] for p in per if ph in p]
            if len(vals) >= 10:
                ph_null[(kind, ph)] = NullModel(pd.DataFrame(vals))
    # phase-maximum statistic: per output, the max over phases of the phase score standardised by the mean and spread
    # of that phase's score on the TRAINING runs. Held-out and candidate runs are then standardised by moments that
    # neither group contributed to, so the two groups stay exchangeable for the permutation test. (Standardising by
    # held-out moments, as before, standardises each held-out run by itself and inflates the false-alarm rate.)
    ph_mom = {}
    for kind, model in (("det", det), ("sig", sigs["dynamic"])):
        per_tr = [phase_scores(model, r, kind) for r in train]
        for ph in phases:
            vals = [p[ph] for p in per_tr if ph in p]
            if len(vals) >= 10 and (kind, ph) in ph_null:
                v = pd.DataFrame(vals)
                ph_mom[(kind, ph)] = (v.mean(axis=0), v.std(axis=0) + 1e-9)

    def phase_max(kind, per_run):
        zs = [(per_run[ph] - ph_mom[(kind, ph)][0]) / ph_mom[(kind, ph)][1] for ph in phases if (kind, ph) in ph_mom and ph in per_run]
        return pd.concat(zs, axis=1).max(axis=1) if zs else pd.Series(np.nan, index=outputs)
    ph_max_null = {}
    for kind, model in (("det", det), ("sig", sigs["dynamic"])):
        ph_max_null[kind] = NullModel(pd.DataFrame([phase_max(kind, phase_scores(model, r, kind)) for r in held]))
    for v in variants_present(ac, mission):
        cand = load(ac, v, mission, range(100, 110))
        if len(cand) < 8:
            continue
        es = b6.effect_size(cand, base_m, outputs)
        is_noop = v == "noop"
        orc = None if is_noop else pd.Series({m: (m in ORACLE.get(fam(v), set())) for m in outputs})
        gated = None if is_noop else (orc & (es > 1.0))
        def score_rec(tag, fl, rsd):
            rec = dict(aircraft=ac, mission=mission, variant=v, mode=tag, n_flagged=int(fl.sum()))
            if is_noop:
                rec["false_alarm"] = float(fl.mean()); return rec
            m = prf(fl, orc); rec.update(recall=m["recall"], precision=m["precision"], auc=auc(rsd, orc))
            if gated.sum() > 0:
                g = prf(fl, gated); rec.update(gated_recall=g["recall"], gated_auc=auc(rsd, gated), n_gated=int(gated.sum()))
            return rec
        # whole-run: detector, signatures, comparators
        rsd, fl = nd.flag_perm(pd.DataFrame([det.score(r) for r in cand]))[:2]; rows.append(score_rec("det_whole", fl, rsd))
        det_flags = fl
        for k, s in sigs.items():
            srsd, sfl = ns[k].flag_perm(pd.DataFrame([s.vsd(r) for r in cand]))[:2]
            rec = score_rec(f"sig_{k}_whole", sfl, srsd)
            rec["agreement"] = agreement(det_flags, sfl)
            rows.append(rec)
        for k, m in comps.items():
            crsd, cfl = nc[k].flag_perm(pd.DataFrame([m.score(r) for r in cand]))[:2]; rows.append(score_rec(f"{k}_whole", cfl, crsd))
        # windowed
        for (kind, W), (model, base_w, nw) in win.items():
            wr, wf = nw.flag_perm(pd.DataFrame([windowed_max(model, r, W, kind, base_w) for r in cand]))[:2]
            rows.append(score_rec(f"{'det' if kind == 'det' else 'sig_dynamic'}_win{W:g}", wf, wr))
        # per phase: flagged if flagged in ANY phase (union), plus per-phase records; explainer correctness per phase
        for kind, model in (("det", det), ("sig", sigs["dynamic"])):
            per = [phase_scores(model, r, kind) for r in cand]
            union = pd.Series(False, index=outputs)
            for ph in phases:
                if (kind, ph) not in ph_null:
                    continue
                vals = [p[ph] for p in per if ph in p]
                if len(vals) < 8:
                    continue
                nph = ph_null[(kind, ph)]
                pr, pf = nph.flag_perm(pd.DataFrame(vals))[:2]
                union |= pf.reindex(outputs).fillna(False)
                rec = score_rec(f"{'det' if kind == 'det' else 'sig_dynamic'}_phase:{ph}", pf, pr)
                if kind == "det" and not is_noop and DRIVING.get(ph):
                    # attribution correctness in this phase with the dynamic signature on the phase slices
                    parts = [r[r["phase"] == ph] for r in cand]; held_parts = [r[r["phase"] == ph] for r in held]
                    corr = []
                    for o in pf[pf].index:
                        a = attribute(sigs["dynamic"], parts, o, held_parts)
                        if a.get("raw") is not None:
                            corr.append(a["raw"] in DRIVING[ph])
                    rec["correctness"] = (float(np.mean(corr)) if corr else np.nan); rec["n_explained"] = len(corr)
                rows.append(rec)
            rows.append(score_rec(f"{'det' if kind == 'det' else 'sig_dynamic'}_phase_any", union, pd.Series(0.0, index=outputs)))
            nm = ph_max_null[kind]
            mr, mf = nm.flag_perm(pd.DataFrame([phase_max(kind, p) for p in per]))[:2]
            rows.append(score_rec(f"{'det' if kind == 'det' else 'sig_dynamic'}_phase_max", mf, mr))
    print(ac, mission, "done", flush=True)
    return rows


def main():
    pairs = []
    for ac in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
        for mission in list(CFG["missions"]) + ["flight_test"]:
            if any(f"_{mission}_" in f for f in os.listdir(os.path.join(RUNS, ac))):
                pairs.append((ac, mission))
    with Pool(int(os.environ.get("WORKERS", 4))) as pool:
        results = pool.map(one_pair, pairs, chunksize=1)
    out = pd.DataFrame([r for rs in results for r in rs])
    out.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "16_tier_b.csv"), index=False)
    pd.set_option("display.width", 220)
    d = out[out.variant != "noop"]
    print("\n== recall / precision / AUC by mode (mean over aircraft, missions, variants); whole vs windows vs phases")
    modes = [m for m in d["mode"].unique() if not m.startswith(("det_phase:", "sig_dynamic_phase:"))]
    print(d[d["mode"].isin(modes)].groupby("mode")[["recall", "precision", "auc", "gated_recall"]].mean().round(3).to_string())
    print("\n== by variant (detector whole vs win30 vs phase_any): recall")
    print(d[d["mode"].isin(["det_whole", "det_win30", "det_win120", "det_phase_any"])].pivot_table(index="variant", columns="mode", values="recall", aggfunc="mean").round(2).to_string())
    print("\n== no-op false alarms by mode:")
    print(out[out.variant == "noop"].groupby("mode")["false_alarm"].mean().round(3).to_string())
    print("\n== attribution correctness per phase (detector flags explained by dynamic signature):")
    ph = out[out["mode"].str.startswith("det_phase:") & out["correctness"].notna()] if "correctness" in out else pd.DataFrame()
    if len(ph):
        print(ph.groupby("mode")["correctness"].mean().round(2).to_string())


if __name__ == "__main__":
    main()

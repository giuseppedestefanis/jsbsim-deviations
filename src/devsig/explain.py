"""Layer 2: explain a flagged output metric with the dynamic signature's violated rules.

Attribution: snapshots of the candidate runs are labelled 'violation' if the signature misclassifies them.
A one-level decision tree on the discretised input features separates violating from non-violating snapshots;
its split feature is the attributed input. The attribution names the raw input, the level, and the timing
(a lag or the time since the input last changed). Correctness = attributed raw input is in the expected set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

DRIVING_INPUT = {  # manoeuvre -> raw inputs that drive it (any lag/since feature of these counts as correct)
    "cruise": set(),
    "throttle_step": {"fcs/throttle-cmd-norm", "fcs/throttle-cmd-norm[1]"},
    "elev_doublet": {"fcs/elevator-cmd-norm", "fcs/elevator-pos-rad", "fcs/elevator-pos-norm"},
    "elev_step": {"fcs/elevator-cmd-norm", "fcs/elevator-pos-rad", "fcs/elevator-pos-norm"},
    "aileron_step": {"fcs/aileron-cmd-norm", "fcs/left-aileron-pos-rad", "fcs/right-aileron-pos-rad", "fcs/left-aileron-pos-norm"},
    "rudder_doublet": {"fcs/rudder-cmd-norm", "fcs/rudder-pos-rad", "fcs/rudder-pos-norm"},
    "bank_hold": {"fcs/aileron-cmd-norm", "fcs/left-aileron-pos-rad", "fcs/right-aileron-pos-rad", "fcs/left-aileron-pos-norm"},
    "flap_extend": {"fcs/flap-cmd-norm", "fcs/flap-pos-deg", "fcs/flap-pos-norm"},
    "engine_cut": {"fcs/throttle-cmd-norm", "fcs/throttle-cmd-norm[1]"},
    "engine_restart": {"propulsion/cutoff_cmd"},
    "engine_stop": {"propulsion/magneto_cmd", "fcs/throttle-cmd-norm"},
}
ELEV = {"fcs/elevator-cmd-norm", "fcs/elevator-pos-rad", "fcs/elevator-pos-norm"}
AIL = {"fcs/aileron-cmd-norm", "fcs/left-aileron-pos-rad", "fcs/right-aileron-pos-rad", "fcs/left-aileron-pos-norm"}
RUD = {"fcs/rudder-cmd-norm", "fcs/rudder-pos-rad", "fcs/rudder-pos-norm"}
THR = {"fcs/throttle-cmd-norm", "fcs/throttle-cmd-norm[1]"}
LONGITUDINAL_OUTPUTS = {"position/h-sl-ft", "position/h-agl-ft", "attitude/theta-rad", "velocities/q-rad_sec", "accelerations/qdot-rad_sec2",
                        "aero/alpha-deg", "velocities/w-fps", "velocities/h-dot-fps", "velocities/u-fps", "velocities/vc-kts", "velocities/vt-fps",
                        "accelerations/udot-ft_sec2", "accelerations/wdot-ft_sec2"}
# manoeuvres with more than one moving input: the driving input depends on the output's axis
DRIVING_BY_AXIS = {
    "climbing_turn": {"longitudinal": ELEV | THR, "lateral": AIL},
    "elev_doublet_distractor": {"longitudinal": ELEV, "lateral": AIL | RUD},
}


def driving_inputs(manoeuvre: str, output: str | None = None) -> set:
    if manoeuvre in DRIVING_BY_AXIS:
        if output is None:
            return set().union(*DRIVING_BY_AXIS[manoeuvre].values())
        return DRIVING_BY_AXIS[manoeuvre]["longitudinal" if output in LONGITUDINAL_OUTPUTS else "lateral"]
    return DRIVING_INPUT.get(manoeuvre, set())


def raw_input(feature: str) -> str:
    return feature.split("@")[0]


def attribute(sig, cand_runs: list[pd.DataFrame], output: str, base_runs: list[pd.DataFrame] | None = None) -> dict:
    """Return the attributed feature for `output` on candidate runs, plus the violation rates.

    Violation labels come from the signature's own misclassifications. To avoid attributing the baseline's
    intrinsic errors, snapshots from base_runs (if given) are included with label 0 weight-balanced, so the
    split finds where candidate violations exceed baseline ones."""
    Xs, ys, ws = [], [], []
    for label, runs in ((1, cand_runs), (0, base_runs or [])):
        for r in runs:
            d = sig.disc.transform(sig._prep(r)[sig.features + sig.outputs])
            X = d[sig.features].to_numpy()
            viol = (sig.trees[output].predict(X) != d[output].to_numpy())
            if label == 1:
                Xs.append(X); ys.append(viol.astype(int)); ws.append(np.ones(len(X)))
            else:
                Xs.append(X[viol]); ys.append(np.zeros(int(viol.sum()), dtype=int)); ws.append(np.ones(int(viol.sum())) * 1.0)
    X = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    rate = float(y[w > 0].mean()) if len(y) else 0.0
    if y.sum() == 0 or y.sum() == len(y):
        return dict(feature=None, raw=None, rate=rate, importance=0.0, level=None, trim_shift=None)
    # Only features that move WITHIN the candidate runs can explain a response to the manoeuvre. A feature that is
    # constant within each run but differs between runs is a trim shift (the faulted model trims differently);
    # it is reported separately, not used for attribution.
    within = np.zeros(len(sig.features))
    for r in cand_runs:
        d = sig._prep(r)[sig.features]
        within += d.std(axis=0).to_numpy()
    dynamic_cols = [i for i, v in enumerate(within) if v > 1e-6]
    trim = trim_shifts(sig, cand_runs, base_runs, [i for i, v in enumerate(within) if v <= 1e-6 and "@" not in sig.features[i]])
    if not dynamic_cols:
        return dict(feature=None, raw=None, rate=rate, importance=0.0, level=None, trim_shift=trim)
    Xd = X[:, dynamic_cols]
    stump = DecisionTreeClassifier(max_depth=1, random_state=0, class_weight="balanced").fit(Xd, y)
    f = int(stump.tree_.feature[0])
    if f < 0:
        return dict(feature=None, raw=None, rate=rate, importance=0.0, level=None, trim_shift=trim)
    f = dynamic_cols[f]
    X = Xd if False else X
    feat = sig.features[f]
    thr = float(stump.tree_.threshold[0])
    # which side has the higher violation share
    left = X[:, f] <= thr
    side = "<=" if y[left].mean() > y[~left].mean() else ">"
    return dict(feature=feat, raw=raw_input(feat), rate=rate, importance=float(stump.tree_.impurity[0] - (stump.tree_.impurity[1] + stump.tree_.impurity[2]) / 2),
                level=f"{side} bin {thr:.1f}", trim_shift=trim)


def trim_shifts(sig, cand_runs, base_runs, const_idx) -> list[str] | None:
    """Inputs constant within every candidate run whose candidate value lies outside the range that the
    baseline runs take for the same input. None when no baseline runs are given (the test cannot be run)."""
    if not base_runs:
        return None
    out = []
    for i in const_idx:
        f = sig.features[i]
        cand_vals = np.array([float(sig._prep(r)[f].iloc[0]) for r in cand_runs])
        base_vals = np.concatenate([sig._prep(r)[f].to_numpy() for r in base_runs])
        lo, hi = float(base_vals.min()), float(base_vals.max())
        tol = 1e-6 * max(1.0, abs(hi), abs(lo))
        if np.median(cand_vals) < lo - tol or np.median(cand_vals) > hi + tol:
            out.append(f)
    return out


def correctness(attr: dict, manoeuvre: str, output: str | None = None) -> bool | None:
    exp = driving_inputs(manoeuvre, output)
    if not exp or attr.get("raw") is None:
        return None
    return attr["raw"] in exp


def agreement(detector_flags: pd.Series, signature_flags: pd.Series) -> float:
    """Share of detector-flagged outputs that the signature also flags at its own calibrated threshold."""
    idx = detector_flags[detector_flags].index
    if len(idx) == 0:
        return float("nan")
    return float(signature_flags.reindex(idx).fillna(False).mean())


def agreement_null_median(detector_flags: pd.Series, signature_rsd: pd.Series, signature_null_median: pd.Series) -> float:
    """Earlier, weaker criterion: fraction of detector-flagged metrics whose signature RSD exceeds the null median."""
    idx = detector_flags[detector_flags].index
    if len(idx) == 0:
        return float("nan")
    return float((signature_rsd[idx] > signature_null_median[idx]).mean())

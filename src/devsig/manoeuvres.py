"""Tier A: QTG-style open-loop manoeuvre battery driven from Python on top of a trimmed state."""
from __future__ import annotations

import json
import math
import os
import zlib

import numpy as np
import pandas as pd

from . import variation
from .run import DEFAULT_PROPS
from .tier_a import fly, load_and_trim, settle_trim

PROPS = DEFAULT_PROPS

# name -> (duration_s, list of (t, prop, delta-or-abs, mode)); mode "delta" adds to trimmed value, "abs" sets.
BATTERY = {
    "cruise":        (60.0, []),
    "throttle_step": (60.0, [(2.0, "fcs/throttle-cmd-norm", +0.2, "delta"), (30.0, "fcs/throttle-cmd-norm", 0.0, "delta")]),
    "elev_doublet":  (30.0, [(2.0, "fcs/elevator-cmd-norm", +0.1, "delta"), (3.0, "fcs/elevator-cmd-norm", -0.1, "delta"), (4.0, "fcs/elevator-cmd-norm", 0.0, "delta")]),
    "elev_step":     (60.0, [(2.0, "fcs/elevator-cmd-norm", -0.1, "delta"), (5.0, "fcs/elevator-cmd-norm", 0.0, "delta")]),
    "aileron_step":  (30.0, [(2.0, "fcs/aileron-cmd-norm", +0.2, "delta"), (5.0, "fcs/aileron-cmd-norm", 0.0, "delta")]),
    "rudder_doublet": (30.0, [(2.0, "fcs/rudder-cmd-norm", +0.2, "delta"), (3.0, "fcs/rudder-cmd-norm", -0.2, "delta"), (4.0, "fcs/rudder-cmd-norm", 0.0, "delta")]),
    "bank_hold":     (45.0, [(2.0, "fcs/aileron-cmd-norm", +0.15, "delta"), (6.0, "fcs/aileron-cmd-norm", 0.0, "delta")]),
    "flap_extend":   (40.0, [(2.0, "fcs/flap-cmd-norm", 0.5, "abs")]),
    "engine_cut":    (40.0, [(2.0, "fcs/throttle-cmd-norm[0]", 0.0, "abs")]),
    # multi-input manoeuvre with correlated controls: power up, roll in, pull, roll out, release
    "climbing_turn": (45.0, [(2.0, "fcs/throttle-cmd-norm", +0.2, "delta"), (2.5, "fcs/aileron-cmd-norm", +0.15, "delta"),
                             (3.0, "fcs/elevator-cmd-norm", -0.08, "delta"), (8.0, "fcs/aileron-cmd-norm", 0.0, "delta"),
                             (12.0, "fcs/elevator-cmd-norm", 0.0, "delta"), (30.0, "fcs/throttle-cmd-norm", 0.0, "delta")]),
    # elevator doublet followed by unrelated (distractor) aileron and rudder pulses
    "elev_doublet_distractor": (40.0, [(2.0, "fcs/elevator-cmd-norm", +0.1, "delta"), (3.0, "fcs/elevator-cmd-norm", -0.1, "delta"),
                                       (4.0, "fcs/elevator-cmd-norm", 0.0, "delta"), (10.0, "fcs/aileron-cmd-norm", +0.1, "delta"),
                                       (11.0, "fcs/aileron-cmd-norm", 0.0, "delta"), (16.0, "fcs/rudder-cmd-norm", +0.1, "delta"),
                                       (17.0, "fcs/rudder-cmd-norm", 0.0, "delta")]),
    # coverage-driven test (jets with two or more engines): shut engine 0 down, spool it to a stop, release the cutoff
    "engine_restart": (60.0, [(1.0, "propulsion/active_engine", 0.0, "abs"), (2.0, "propulsion/cutoff_cmd", 1.0, "abs"),
                              (4.0, "propulsion/engine[0]/n1", 0.0, "abs"), (4.0, "propulsion/engine[0]/n2", 0.0, "abs"),
                              (20.0, "propulsion/cutoff_cmd", 0.0, "abs")]),
    # coverage-driven test (piston aircraft): ignition off and throttle closed, engine stops, 40 s glide
    "engine_stop": (40.0, [(2.0, "propulsion/magneto_cmd", 0.0, "abs"), (2.0, "fcs/throttle-cmd-norm", 0.0, "abs")]),
}
JET_CLASSES = ("airliner", "fighter", "trainer", "bizjet")
PISTON_CLASSES = ("light", "vintage", "cub")


def load_tier_a_config(path=None):
    path = path or os.path.join(os.path.dirname(__file__), "..", "..", "config", "tier_a_aircraft.json")
    with open(path) as f:
        return json.load(f)


# MIL-F-8785C probability-of-exceedance level: 3 (moderate; non-zero up to 35,000 ft) is the default study
# condition; DEVSIG_TURB_SEVERITY=0 gives no turbulence (every other benign variation kept), the qualification condition
TURB_SEVERITY = int(os.environ.get("DEVSIG_TURB_SEVERITY", "3"))
TURB_WIND20_FPS = 15.0  # only used by the model below 2,000 ft, which the flights here do not reach


def enable_turbulence(fdm, seed: int):
    """Switch on MIL-F-8785C turbulence at exceedance level TURB_SEVERITY after the trim (the aircraft is
    trimmed without turbulence; the gusts act during the recorded run). Level 1 is zero above 7,500 ft and level 2
    above 15,000 ft, so level 3 is used at every altitude flown here. The gust sequence comes from JSBSim's
    own generator, seeded with the run seed, so it differs between runs."""
    pm = fdm.get_property_manager()
    if pm.hasNode("atmosphere/turb-type") and TURB_SEVERITY > 0:
        fdm["atmosphere/turb-type"] = 3  # MIL-STD-1797 (Milspec)
        fdm["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = TURB_WIND20_FPS
        fdm["atmosphere/turbulence/milspec/severity"] = TURB_SEVERITY
    fdm["simulation/randomseed"] = seed
    if pm.hasNode("atmosphere/randomseed"):
        fdm["atmosphere/randomseed"] = seed


def salt(*names: str) -> list[int]:
    """Stable integers from names (aircraft, manoeuvre), so that the benign draws of a seed differ between
    test cases: a chance imbalance of one seed set then does not repeat on every aircraft."""
    return [zlib.crc32(n.encode()) for n in names]


def trimmed_fdm(aircraft: str, cfg: dict, seed: int, root_dir=None, ic_jitter=True, test_case: str = ""):
    """Trim with benign variation applied (fuel, payload, wind, turbulence, IC jitter). Returns (fdm, ok, msg, var)."""
    # independent random streams for the variation draw and the initial-condition jitter, specific to the
    # run seed, the aircraft and the test case
    s_var, s_ic = np.random.SeedSequence([seed] + salt(aircraft, test_case)).spawn(2)
    rng = np.random.default_rng(s_ic)
    v = variation.draw(seed, rng=np.random.default_rng(s_var))
    h = cfg["h_ft"] + (rng.uniform(-200, 200) if ic_jitter else 0.0)
    vc = cfg["vc_kts"] + (rng.uniform(-5, 5) if ic_jitter else 0.0)

    def post_ic(fdm):
        variation.apply(fdm, v)
    fn = load_and_trim if cfg["trim"] == "analytic" else settle_trim
    fdm, ok, msg = fn(aircraft, h, vc, root_dir=root_dir, post_ic=post_ic)
    if not ok and fn is load_and_trim and not msg.startswith(("load:", "no engine")):
        fdm, ok, msg = settle_trim(aircraft, h, vc, root_dir=root_dir, post_ic=post_ic)
        msg = "fallback " + msg
    if ok:
        enable_turbulence(fdm, seed)
    var = dict(v.as_dict(), h_ft=h, vc_kts=vc, turb_severity=TURB_SEVERITY, randomseed=seed)
    return fdm, ok, msg, var


AMP_SCALE = {"fighter": 0.4, "trainer": 0.6}  # smaller control inputs for fast aircraft (sustained bank otherwise diverges)


def run_manoeuvre(fdm, name: str, seed: int, sample_hz: float = 10.0, jitter=True, cls: str = "", aircraft: str = "") -> pd.DataFrame:
    dur, events = BATTERY[name]
    scale = AMP_SCALE.get(cls, 1.0)
    # seeded by the run seed, the aircraft and the manoeuvre: stable across processes and sessions
    rng = np.random.default_rng([seed, list(BATTERY).index(name)] + salt(aircraft))
    pm = fdm.get_property_manager()
    inputs = []
    base = {}
    for t, prop, val, mode in events:
        if not pm.hasNode(prop):
            continue
        if prop not in base:
            base[prop] = fdm[prop]
        tt = t + (rng.uniform(-0.3, 0.3) if jitter and t > 0 else 0.0)
        amp = val * (rng.uniform(0.9, 1.1) if jitter and val != 0 and prop.startswith("fcs/") else 1.0) * (scale if mode == "delta" else 1.0)
        target = (base[prop] + amp) if mode == "delta" else amp
        inputs.append((max(0.0, tt), prop, float(np.clip(target, -1.0, 1.0))))
    return fly(fdm, dur, sample_hz=sample_hz, inputs=inputs, props=PROPS)


def tier_a_run(aircraft: str, cfg: dict, manoeuvre: str, seed: int, root_dir=None) -> tuple[pd.DataFrame | None, str, dict | None]:
    fdm, ok, msg, var = trimmed_fdm(aircraft, cfg, seed, root_dir=root_dir, test_case=manoeuvre)
    if not ok:
        return None, f"trim failed: {msg}", None
    if manoeuvre == "engine_cut" and fdm.get_propulsion().get_num_engines() < 2:
        return None, "single engine: skip engine_cut", None
    if manoeuvre == "engine_restart" and (cfg.get("cls") not in JET_CLASSES or fdm.get_propulsion().get_num_engines() < 2):
        return None, "engine_restart needs a multi-engine jet: skip", None
    if manoeuvre == "engine_stop" and (cfg.get("cls") not in PISTON_CLASSES or not fdm.get_property_manager().hasNode("propulsion/magneto_cmd")):
        return None, "engine_stop needs a piston aircraft: skip", None
    df = run_manoeuvre(fdm, manoeuvre, seed, cls=cfg.get("cls", ""), aircraft=aircraft)
    if df.isna().any().any() or (df["position/h-agl-ft"] <= 0).any():
        return None, "NaN or ground contact during manoeuvre", None
    return df, "ok", var

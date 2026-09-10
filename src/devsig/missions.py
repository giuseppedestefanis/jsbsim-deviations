"""Tier B: closed-loop missions with phases, driven through the aircraft's autopilot from a trimmed airborne state.

A mission is a list of (t_s, phase_name, {property: value}) entries: at time t the phase starts and the
autopilot setpoints are changed. The recorded run carries a 'phase' column. Missions are the same for every
aircraft with the shared autopilot (altitude hold, heading hold, airspeed hold when present); setpoints are
relative to the trimmed state so the same mission fits a Cessna and a business jet.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .manoeuvres import salt, trimmed_fdm
from .run import DEFAULT_PROPS

PROPS = list(dict.fromkeys(DEFAULT_PROPS + ["ap/altitude_setpoint", "ap/heading_setpoint", "ap/airspeed_setpoint"]))

# (t_s, phase, relative setpoint changes). "alt" in ft relative to trim altitude, "hdg" in deg relative to trim
# heading, "spd" in kt relative to trim airspeed, "flap" absolute, "gear" absolute.
MISSIONS = {
    "climb_cruise_descend": (720.0, [
        (0.0,   "settle",   {}),
        (30.0,  "climb",    {"alt": +2000}),
        (180.0, "turn",     {"hdg": +90}),
        (270.0, "cruise",   {}),
        (360.0, "slow",     {"spd": -20}),
        (450.0, "descend",  {"alt": -3000}),
        (600.0, "turnback", {"hdg": -90}),
        (690.0, "level",    {}),
    ]),
    "approach": (480.0, [
        (0.0,   "settle",   {}),
        (30.0,  "descend",  {"alt": -2500, "spd": -15}),
        (210.0, "config",   {"flap": 0.5, "spd": -15}),
        (330.0, "gear",     {"gear": 1.0}),
        (400.0, "final",    {"alt": -800}),
    ]),
}


def hand_over_trim(fdm):
    """Move the trimmed manual control positions into the trim channels so the autopilot commands add to a
    neutral stick: elevator-cmd -> pitch-trim-cmd, aileron-cmd -> roll-trim-cmd, rudder-cmd -> yaw-trim-cmd."""
    pm = fdm.get_property_manager()
    for cmd, trim in (("fcs/elevator-cmd-norm", "fcs/pitch-trim-cmd-norm"), ("fcs/aileron-cmd-norm", "fcs/roll-trim-cmd-norm"),
                      ("fcs/rudder-cmd-norm", "fcs/yaw-trim-cmd-norm")):
        if pm.hasNode(cmd) and pm.hasNode(trim):
            fdm[trim] = fdm[trim] + fdm[cmd]
            fdm[cmd] = 0.0


class Autothrottle:
    """PI controller on calibrated airspeed driving every engine's throttle command (the pilot's input)."""
    def __init__(self, fdm, kp=0.02, ki=0.004, lo=0.05, hi=1.0):
        self.n = fdm.get_propulsion().get_num_engines()
        self.kp, self.ki, self.lo, self.hi = kp, ki, lo, hi
        self.base = fdm["fcs/throttle-cmd-norm"]; self.i = 0.0

    def step(self, fdm, spd_setpoint, dt):
        err = spd_setpoint - fdm["velocities/vc-kts"]
        self.i = max(-0.4, min(0.4, self.i + self.ki * err * dt))
        thr = max(self.lo, min(self.hi, self.base + self.kp * err + self.i))
        for k in range(self.n):
            fdm[f"fcs/throttle-cmd-norm[{k}]"] = thr


def engage_autopilot(fdm, alt_ft, hdg_deg, spd_kts):
    pm = fdm.get_property_manager()
    hand_over_trim(fdm)
    fdm["ap/altitude_setpoint"] = alt_ft
    fdm["ap/heading_setpoint"] = hdg_deg
    if pm.hasNode("ap/airspeed_setpoint"):
        fdm["ap/airspeed_setpoint"] = spd_kts
    if pm.hasNode("ap/attitude_hold"):
        fdm["ap/attitude_hold"] = 0
    fdm["ap/altitude_hold"] = 1
    fdm["ap/heading_hold"] = 1
    if pm.hasNode("ap/heading-setpoint-select"):
        fdm["ap/heading-setpoint-select"] = 0
    if pm.hasNode("ap/airspeed_hold"):
        fdm["ap/airspeed_hold"] = 1


def run_mission(fdm, mission: str, seed: int, sample_hz: float = 10.0, jitter: bool = True, aircraft: str = "") -> pd.DataFrame:
    dur, plan = MISSIONS[mission]
    rng = np.random.default_rng([seed, 7] + salt(aircraft, mission))
    pm = fdm.get_property_manager()
    alt0 = fdm["position/h-sl-ft"]; hdg0 = math.degrees(fdm["attitude/psi-rad"]) % 360.0; spd0 = fdm["velocities/vc-kts"]
    engage_autopilot(fdm, alt0, hdg0, spd0)
    at = Autothrottle(fdm)
    avail = [p for p in PROPS if pm.hasNode(p)]
    dt = fdm.get_delta_t(); every = max(1, int(round(1.0 / (sample_hz * dt))))
    alt, hdg, spd = alt0, hdg0, spd0
    events = []
    for t, name, ch in plan:
        tt = t + (rng.uniform(-1.0, 1.0) if jitter and t > 0 else 0.0)
        events.append((tt, name, ch))
    events.sort(key=lambda e: e[0])
    rows, step, phase = [], 0, plan[0][1]
    t0 = fdm.get_sim_time()
    pending = list(events)
    while fdm.get_sim_time() - t0 < dur:
        t = fdm.get_sim_time() - t0
        while pending and pending[0][0] <= t:
            _, phase, ch = pending.pop(0)
            if "alt" in ch:
                alt = alt + ch["alt"] * (rng.uniform(0.95, 1.05) if jitter else 1.0); fdm["ap/altitude_setpoint"] = alt
            if "hdg" in ch:
                hdg = (hdg + ch["hdg"]) % 360.0; fdm["ap/heading_setpoint"] = hdg
            if "spd" in ch:
                spd = spd + ch["spd"]
                if pm.hasNode("ap/airspeed_setpoint"):
                    fdm["ap/airspeed_setpoint"] = spd
            if "flap" in ch and pm.hasNode("fcs/flap-cmd-norm"):
                fdm["fcs/flap-cmd-norm"] = ch["flap"]
            if "gear" in ch and pm.hasNode("gear/gear-cmd-norm"):
                fdm["gear/gear-cmd-norm"] = ch["gear"]
        at.step(fdm, spd, dt)
        if step % every == 0:
            rows.append([fdm.get_sim_time(), phase] + [fdm[p] for p in avail])
        if not fdm.run():
            break
        step += 1
    return pd.DataFrame(rows, columns=["t", "phase"] + avail)


def tier_b_run(aircraft: str, cfg: dict, mission: str, seed: int, root_dir=None):
    fdm, ok, msg, var = trimmed_fdm(aircraft, cfg, seed, root_dir=root_dir, test_case=mission)
    if not ok:
        return None, f"trim failed: {msg}", None
    df = run_mission(fdm, mission, seed, aircraft=aircraft)
    if df.isna().any().any() or (df["position/h-agl-ft"] <= 0).any():
        return None, "NaN or ground contact during mission", None
    return df, "ok", var

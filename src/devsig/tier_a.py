"""Tier A: trim a stock aircraft from Python and fly open-loop manoeuvres."""
from __future__ import annotations

import contextlib
import io
import math

import jsbsim
import pandas as pd

from .run import DEFAULT_PROPS

# Candidate trim points per rough class: (altitude ft, calibrated airspeed kt).
TRIM_GRID = {
    "light":   [(5000, 100), (5000, 120), (3000, 90), (5000, 140), (8000, 110)],
    "turboprop": [(10000, 180), (8000, 150), (12000, 220), (5000, 130)],
    "bizjet":  [(20000, 250), (30000, 220), (15000, 280), (10000, 250)],
    "airliner": [(20000, 280), (30000, 260), (15000, 300), (10000, 250), (25000, 300)],
    "fighter": [(20000, 350), (15000, 400), (25000, 450), (10000, 300), (30000, 500)],
    "vintage": [(3000, 80), (3000, 100), (5000, 120), (2000, 60)],
    "transport": [(15000, 250), (10000, 220), (20000, 280), (5000, 200)],
    "other":   [(10000, 200), (5000, 150), (20000, 300), (3000, 100), (30000, 400)],
    "trainer": [(10000, 200), (8000, 180), (15000, 220), (5000, 160)],
    "cub":     [(2000, 60), (2000, 70), (3000, 75), (1500, 55)],
    "bomber":  [(5000, 150), (8000, 160), (5000, 130), (10000, 170)],
}

AIRCRAFT_CLASS = {
    "737": "airliner", "787-8": "airliner", "A320": "airliner", "B747": "airliner", "MD11": "airliner",
    "fokker100": "airliner", "fokker50": "turboprop", "Concorde": "airliner",
    "C130": "transport", "DHC6": "light", "L410": "turboprop", "pc7": "turboprop", "t6texan2": "turboprop",
    "OV10": "turboprop", "global5000": "bizjet",
    "c172x": "light", "c172p": "light", "c172r": "light", "c182": "light", "c310": "light", "pa28": "light",
    "L17": "light", "J3Cub": "cub",
    "f15": "fighter", "f16": "fighter", "f22": "fighter", "F4N": "fighter", "f104": "fighter", "F80C": "fighter",
    "A4": "fighter", "T37": "trainer", "T38": "fighter", "X15": "other", "XB-70": "other",
    "p51d": "vintage", "B17": "bomber", "Camel": "vintage", "dr1": "vintage", "Short_S23": "vintage",
    "Boeing314": "vintage", "wrightFlyer1903": "vintage", "SGS": "other", "sgs126": "other", "sgs233": "other",
    "minisgs": "other", "paraglider": "other",
}

FIXED_WING_EXCLUDE = {"ball", "ballx", "blank", "aircraft_template.xml", "ah1s", "F450", "J246", "mk82",
                      "pogo-jsbsim", "Pterosaur", "Shuttle", "Submarine_Scout", "weather-balloon", "ZLT-NT",
                      "x24b", "wrightFlyer1903"}


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def load_and_trim(aircraft: str, h_ft: float, vc_kts: float, root_dir: str | None = None,
                  aircraft_path: str | None = None, overrides: dict | None = None, post_ic=None):
    """Load model, set ICs, start engines, trim. Returns (fdm, ok, msg)."""
    with _quiet():
        fdm = jsbsim.FGFDMExec(root_dir)
        fdm.set_debug_level(0)
        if aircraft_path:
            fdm.set_aircraft_path(aircraft_path)
        try:
            fdm.load_model(aircraft)
        except Exception as e:  # noqa: BLE001
            return None, False, f"load: {e}"
        fdm["ic/h-sl-ft"] = h_ft
        fdm["ic/vc-kts"] = vc_kts
        fdm["ic/gamma-deg"] = 0.0
        fdm["ic/psi-true-deg"] = 90.0
        fdm["ic/lat-geod-deg"] = 29.6
        fdm["ic/long-gc-deg"] = -95.16
        fdm["ic/terrain-elevation-ft"] = 0.0
        for k, v in (overrides or {}).items():
            fdm[k] = v
        try:
            n_eng = fdm.get_propulsion().get_num_engines()
            if n_eng == 0:
                return fdm, False, "no engine (glider)"
            pm = fdm.get_property_manager()
            for i in range(n_eng):
                for k in (f"fcs/mixture-cmd-norm[{i}]", f"fcs/advance-cmd-norm[{i}]"):
                    if pm.hasNode(k):
                        fdm[k] = 1.0
            if pm.hasNode("propulsion/magneto_cmd"):
                fdm["propulsion/magneto_cmd"] = 3
            fdm["propulsion/set-running"] = -1
            fdm.run_ic()
            fdm["gear/gear-cmd-norm"] = 0.0
            if post_ic is not None:
                post_ic(fdm)
            fdm["simulation/do_simple_trim"] = 1
        except Exception as e:  # noqa: BLE001
            return fdm, False, f"trim: {type(e).__name__}: {str(e)[:80]}"
    if fdm["simulation/trim-completed"] != 1:
        return fdm, False, "trim: not completed"
    return fdm, True, "ok"


def fly(fdm, seconds: float, sample_hz: float = 10.0, inputs=None, props=None) -> pd.DataFrame:
    """Advance the sim `seconds`, applying timed input sets, sampling props.

    inputs: list of (t_rel_s, property, value); applied when sim time passes t0 + t_rel.
    """
    props = props or DEFAULT_PROPS
    pm = fdm.get_property_manager()
    avail = [p for p in props if pm.hasNode(p)]
    dt = fdm.get_delta_t()
    every = max(1, int(round(1.0 / (sample_hz * dt))))
    t0 = fdm.get_sim_time()
    pending = sorted(inputs or [], key=lambda x: x[0])
    rows, step = [], 0
    with _quiet():
        while fdm.get_sim_time() - t0 < seconds:
            t = fdm.get_sim_time() - t0
            while pending and pending[0][0] <= t:
                _, k, v = pending.pop(0)
                fdm[k] = v
            if step % every == 0:
                rows.append([fdm.get_sim_time()] + [fdm[p] for p in avail])
            if not fdm.run():
                break
            step += 1
    return pd.DataFrame(rows, columns=["t"] + avail)


def stable(df: pd.DataFrame, dh_max=500.0, dv_max=15.0, phi_max=15.0) -> tuple[bool, str]:
    if df.isna().any().any():
        return False, "NaN"
    if (df["position/h-agl-ft"] <= 0).any():
        return False, "ground contact"
    dh = df["position/h-sl-ft"].iloc[-1] - df["position/h-sl-ft"].iloc[0]
    dv = df["velocities/vc-kts"].iloc[-1] - df["velocities/vc-kts"].iloc[0]
    phi = math.degrees(df["attitude/phi-rad"].abs().max())
    if abs(dh) > dh_max:
        return False, f"altitude drift {dh:.0f} ft"
    if abs(dv) > dv_max:
        return False, f"speed drift {dv:.1f} kt"
    if phi > phi_max:
        return False, f"bank {phi:.0f} deg"
    return True, f"dh={dh:.0f} dv={dv:.1f} phi={phi:.1f}"


def doublet(fdm, prop="fcs/elevator-cmd-norm", amp=0.2, hold=1.0, seconds=20.0, sample_hz=10.0):
    base = fdm[prop]
    inputs = [(1.0, prop, base + amp), (1.0 + hold, prop, base - amp), (1.0 + 2 * hold, prop, base)]
    return fly(fdm, seconds, sample_hz, inputs)


def start_engines(fdm, throttle: float = 0.7):
    """Start all engines in a way that works for piston, turboprop, and turbine models."""
    pm = fdm.get_property_manager()
    n = fdm.get_propulsion().get_num_engines()
    piston = pm.hasNode("propulsion/engine/map-inhg")
    for i in range(n):
        if pm.hasNode(f"fcs/mixture-cmd-norm[{i}]"):
            fdm[f"fcs/mixture-cmd-norm[{i}]"] = 0.8 if piston else 1.0
        if piston and pm.hasNode(f"fcs/advance-cmd-norm[{i}]"):
            fdm[f"fcs/advance-cmd-norm[{i}]"] = 0.8
        if pm.hasNode(f"fcs/throttle-cmd-norm[{i}]"):
            fdm[f"fcs/throttle-cmd-norm[{i}]"] = throttle
    if pm.hasNode("propulsion/magneto_cmd"):
        fdm["propulsion/magneto_cmd"] = 3
    if pm.hasNode("propulsion/starter_cmd"):
        fdm["propulsion/starter_cmd"] = 1
    fdm["propulsion/set-running"] = -1


def _advance(fdm, seconds):
    t0 = fdm.get_sim_time()
    while fdm.get_sim_time() - t0 < seconds:
        fdm.run()


def settle_trim(aircraft: str, h_ft: float, vc_kts: float, root_dir=None, aircraft_path=None,
                overrides=None, settle_s: float = 120.0, check_s: float = 20.0, try_analytic: bool = True, post_ic=None):
    """Auto-trim by closed-loop settling, then freeze the controls.

    Altitude is held with the elevator (PD on altitude error with pitch-rate damping), airspeed with the
    throttle (PI), wings level with the aileron, sideslip with the rudder. After `settle_s` seconds the last
    `check_s` seconds are checked for drift; if small, the controls are frozen at their settled values and,
    if `try_analytic`, JSBSim's own trimmer is attempted from this near-trim state (used if it succeeds).
    Returns (fdm, ok, msg). Works for piston, turboprop, and turbine models alike.
    """
    with _quiet():
        fdm = jsbsim.FGFDMExec(root_dir)
        fdm.set_debug_level(0)
        if aircraft_path:
            fdm.set_aircraft_path(aircraft_path)
        try:
            fdm.load_model(aircraft)
        except Exception as e:  # noqa: BLE001
            return None, False, f"load: {e}"
        n = fdm.get_propulsion().get_num_engines()
        if n == 0:
            return fdm, False, "no engine (glider)"
        fdm["ic/h-sl-ft"] = h_ft
        fdm["ic/vc-kts"] = vc_kts
        fdm["ic/gamma-deg"] = 0.0
        fdm["ic/psi-true-deg"] = 90.0
        fdm["ic/lat-geod-deg"] = 29.6
        fdm["ic/long-gc-deg"] = -95.16
        fdm["ic/terrain-elevation-ft"] = 0.0
        for k, v in (overrides or {}).items():
            fdm[k] = v
        try:
            fdm["propulsion/set-running"] = -1
            fdm.run_ic()
            fdm["gear/gear-cmd-norm"] = 0.0
            if post_ic is not None:
                post_ic(fdm)
            start_engines(fdm, 0.7)
        except Exception as e:  # noqa: BLE001
            return fdm, False, f"start: {type(e).__name__}"
        pm = fdm.get_property_manager()
        has_rudder = pm.hasNode("fcs/rudder-cmd-norm")
        dt = fdm.get_delta_t()
        thr, ele = 0.7, 0.0
        v_int = 0.0
        t0 = fdm.get_sim_time()
        hist = []
        while fdm.get_sim_time() - t0 < settle_s:
            if not fdm.run():
                return fdm, False, "sim stopped during settle"
            h = fdm["position/h-sl-ft"]; hdot = fdm["velocities/h-dot-fps"]
            v = fdm["velocities/vc-kts"]; q = fdm["velocities/q-rad_sec"]
            phi = fdm["attitude/phi-rad"]; p = fdm["velocities/p-rad_sec"]
            if math.isnan(h) or math.isnan(v):
                return fdm, False, "NaN during settle"
            # elevator: negative = nose up in JSBSim convention
            ele_cmd = 0.0015 * (h - h_ft) + 0.02 * hdot + 1.0 * q
            ele = max(-0.6, min(0.6, ele_cmd))
            # throttle: PI on airspeed error
            v_err = vc_kts - v
            v_int = max(-0.5, min(0.5, v_int + 0.002 * v_err * dt))
            thr = max(0.0, min(1.0, 0.7 + 0.01 * v_err + v_int))
            ail = max(-0.5, min(0.5, -1.5 * phi - 0.3 * p))
            fdm["fcs/elevator-cmd-norm"] = ele
            fdm["fcs/aileron-cmd-norm"] = ail
            if has_rudder:
                fdm["fcs/rudder-cmd-norm"] = max(-0.3, min(0.3, -0.02 * fdm["aero/beta-deg"]))
            for i in range(n):
                fdm[f"fcs/throttle-cmd-norm[{i}]"] = thr
            if fdm.get_sim_time() - t0 > settle_s - check_s:
                hist.append((h, v, phi))
        hs = [x[0] for x in hist]; vs = [x[1] for x in hist]; phis = [abs(x[2]) for x in hist]
        dh = max(hs) - min(hs); dv = max(vs) - min(vs); phi_max = math.degrees(max(phis))
        off_h = abs(hist[-1][0] - h_ft); off_v = abs(hist[-1][1] - vc_kts)
        msg = f"settled dh={dh:.0f} dv={dv:.1f} phi={phi_max:.1f} off_h={off_h:.0f} off_v={off_v:.1f} thr={thr:.2f} ele={ele:.3f}"
        if dh > 100 or dv > 5 or phi_max > 10 or off_h > 300 or off_v > 15:
            return fdm, False, "not " + msg
        if try_analytic:
            try:
                fdm["simulation/do_simple_trim"] = 1
                if fdm["simulation/trim-completed"] == 1:
                    return fdm, True, "analytic-after-settle " + msg
            except Exception:  # noqa: BLE001
                pass
        # freeze controls at settled values (already set)
    return fdm, True, msg

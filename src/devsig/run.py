"""Execute one JSBSim scripted run and return sampled properties as a DataFrame."""
from __future__ import annotations

import contextlib
import io
import os
import time
from dataclasses import dataclass, field

import jsbsim
import pandas as pd

DEFAULT_PROPS = [
    "simulation/sim-time-sec",
    "position/h-sl-ft", "position/h-agl-ft", "position/lat-geod-deg", "position/long-gc-deg",
    "attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad",
    "velocities/u-fps", "velocities/v-fps", "velocities/w-fps",
    "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec",
    "velocities/vc-kts", "velocities/vt-fps", "velocities/h-dot-fps",
    "accelerations/pdot-rad_sec2", "accelerations/qdot-rad_sec2", "accelerations/rdot-rad_sec2",
    "accelerations/udot-ft_sec2", "accelerations/vdot-ft_sec2", "accelerations/wdot-ft_sec2",
    "aero/alpha-deg", "aero/beta-deg",
    "fcs/throttle-cmd-norm", "fcs/elevator-cmd-norm", "fcs/aileron-cmd-norm", "fcs/rudder-cmd-norm",
    "fcs/flap-cmd-norm", "fcs/elevator-pos-rad", "fcs/left-aileron-pos-rad", "fcs/rudder-pos-rad",
    "fcs/flap-pos-deg", "gear/gear-cmd-norm", "gear/gear-pos-norm",
    "fcs/throttle-cmd-norm[1]", "fcs/pitch-trim-cmd-norm", "fcs/right-aileron-pos-rad", "fcs/elevator-pos-norm",
    "fcs/rudder-pos-norm", "fcs/flap-pos-norm", "fcs/left-aileron-pos-norm",
    "ap/altitude_setpoint", "ap/heading_setpoint",
    "propulsion/cutoff_cmd", "propulsion/magneto_cmd",
]


@dataclass
class RunResult:
    script: str
    aircraft: str
    ok: bool
    sim_end_s: float
    wall_s: float
    steps: int
    error: str = ""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_script(script_rel: str, sample_hz: float = 10.0, max_wall_s: float = 120.0,
               props: list[str] | None = None, root_dir: str | None = None,
               property_overrides: dict[str, float] | None = None,
               aircraft_path: str | None = None, quiet: bool = True, post_ic=None) -> RunResult:
    """Run a JSBSim script to completion (or until max_wall_s), sampling `props` at sample_hz.

    property_overrides are applied after run_ic(), e.g. benign wind or fuel changes.
    aircraft_path lets a mutated aircraft directory replace the stock one.
    """
    props = props or DEFAULT_PROPS
    t0 = time.time()
    sink = io.StringIO() if quiet else None
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        fdm = jsbsim.FGFDMExec(root_dir)
        fdm.set_debug_level(0)
        if aircraft_path:
            fdm.set_aircraft_path(aircraft_path)
        try:
            fdm.load_script(script_rel)
        except Exception as e:  # noqa: BLE001
            return RunResult(script_rel, "?", False, 0.0, time.time() - t0, 0, f"load: {e}")
        fdm.disable_output()
        aircraft = fdm.get_model_name()
        try:
            fdm.run_ic()
        except Exception as e:  # noqa: BLE001
            return RunResult(script_rel, aircraft, False, 0.0, time.time() - t0, 0, f"run_ic: {e}")
        for k, v in (property_overrides or {}).items():
            fdm[k] = v
        if post_ic is not None:
            post_ic(fdm)
        dt = fdm.get_delta_t()
        every = max(1, int(round(1.0 / (sample_hz * dt))))
        rows, step, err = [], 0, ""
        avail = [p for p in props if fdm.get_property_manager().hasNode(p)]
        try:
            while True:
                if step % every == 0:
                    rows.append([fdm[p] for p in avail])
                if not fdm.run():
                    break
                step += 1
                if step % 5000 == 0 and time.time() - t0 > max_wall_s:
                    err = "wall-timeout"
                    break
        except Exception as e:  # noqa: BLE001
            err = f"run: {e}"
        sim_end = fdm.get_sim_time()
    df = pd.DataFrame(rows, columns=avail)
    ok = err == "" and not df.isna().any().any()
    if err == "" and df.isna().any().any():
        err = "NaN in output"
    return RunResult(script_rel, aircraft, ok, sim_end, time.time() - t0, step, err, df)


def load_run(path):
    """Read a stored run. The heading angle (attitude/psi-rad) is recorded in [0, 2 pi) and jumps by 2 pi when the
    aircraft turns through north; it is unwrapped here so that distances between runs measure real differences."""
    import numpy as np
    import pandas as pd
    df = pd.read_parquet(path)
    if "attitude/psi-rad" in df.columns:
        df["attitude/psi-rad"] = np.unwrap(df["attitude/psi-rad"].to_numpy())
    return df

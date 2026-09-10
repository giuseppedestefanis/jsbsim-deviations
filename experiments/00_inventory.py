"""Phase 0: run every stock script once and record what flies unattended."""
import glob
import os
import sys

import jsbsim
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from devsig.run import run_script  # noqa: E402

root = jsbsim.get_default_root_dir()
scripts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "scripts", "*.xml")))
skip = {"plotfile.xml", "unitconversions.xml", "kml_output.xml"}
recs = []
for s in scripts:
    if s in skip:
        continue
    r = run_script(f"scripts/{s}", sample_hz=1.0, max_wall_s=90.0)
    d = r.df
    rec = dict(script=s, aircraft=r.aircraft, ok=r.ok, error=r.error, sim_end_s=round(r.sim_end_s, 1),
               wall_s=round(r.wall_s, 1), samples=len(d))
    if len(d):
        for col, name in [("position/h-sl-ft", "h_max_ft"), ("velocities/vc-kts", "vc_max_kts")]:
            if col in d:
                rec[name] = round(float(d[col].max()), 0)
        if "position/h-agl-ft" in d:
            rec["h_agl_end_ft"] = round(float(d["position/h-agl-ft"].iloc[-1]), 0)
        if "position/lat-geod-deg" in d:
            rec["lat_span_deg"] = round(float(d["position/lat-geod-deg"].max() - d["position/lat-geod-deg"].min()), 3)
    recs.append(rec)
    print(f"{s:40s} {r.aircraft:14s} ok={r.ok!s:5s} end={r.sim_end_s:7.1f}s wall={r.wall_s:5.1f}s {r.error}", flush=True)

out = pd.DataFrame(recs)
os.makedirs("results", exist_ok=True)
out.to_csv("results/00_inventory.csv", index=False)
print(out.to_string())

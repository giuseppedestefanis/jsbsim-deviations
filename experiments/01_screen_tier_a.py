"""Phase 1: screen every fixed-wing stock model for Tier A (trim, hands-off, doublet)."""
import json
import os
import subprocess
import sys

import jsbsim
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
from devsig.tier_a import AIRCRAFT_CLASS, FIXED_WING_EXCLUDE, TRIM_GRID  # noqa: E402

WORKER = r'''
import json, sys, math
sys.path.insert(0, %r)
from devsig.tier_a import load_and_trim, settle_trim, fly, stable, doublet, TRIM_GRID, AIRCRAFT_CLASS
ac = sys.argv[1]
cls = AIRCRAFT_CLASS.get(ac, "other")
out = dict(aircraft=ac, cls=cls, trim_ok=False, trim_method="", trim_msg="", h_ft=None, vc_kts=None,
           handsoff_ok=False, handsoff_msg="", doublet_ok=False, doublet_msg="", n_engines=None)
done = False
for method in ("analytic", "settle"):
    if done:
        break
    for h, v in TRIM_GRID[cls]:
        fdm, ok, msg = (load_and_trim if method == "analytic" else settle_trim)(ac, h, v)
        out["trim_msg"] = f"{method}: {msg}"
        if fdm is not None:
            out["n_engines"] = fdm.get_propulsion().get_num_engines()
        if msg.startswith(("load:", "no engine")):
            done = True
            break
        if not ok:
            continue
        df = fly(fdm, 60.0, sample_hz=2.0)
        s_ok, s_msg = stable(df)
        if not s_ok:
            out["trim_msg"] = f"{method}: trimmed at {h}/{v} but " + s_msg
            continue
        out.update(trim_ok=True, trim_method=method, h_ft=h, vc_kts=v, handsoff_ok=True, handsoff_msg=s_msg)
        done = True
        break
if out["trim_ok"]:
    d = doublet(fdm, amp=0.1)
    d_ok, d_msg = stable(d, dh_max=1500, dv_max=40, phi_max=30)
    out.update(doublet_ok=d_ok, doublet_msg=d_msg,
               doublet_q_max=float(d["velocities/q-rad_sec"].abs().max()) if "velocities/q-rad_sec" in d else None)
print(json.dumps(out))
'''

root = jsbsim.get_default_root_dir()
aircraft = sorted(a for a in os.listdir(os.path.join(root, "aircraft"))
                  if os.path.isdir(os.path.join(root, "aircraft", a)) and a not in FIXED_WING_EXCLUDE)
recs = []
for ac in aircraft:
    try:
        r = subprocess.run([sys.executable, "-c", WORKER % os.path.abspath(SRC), ac],
                           capture_output=True, text=True, timeout=600)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        rec = json.loads(line[-1]) if line else dict(aircraft=ac, trim_msg=f"worker failed rc={r.returncode}: {r.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        rec = dict(aircraft=ac, trim_msg="worker timeout")
    recs.append(rec)
    print(f"{ac:16s} trim={rec.get('trim_ok')!s:5s} {rec.get('trim_method','')[:8]:8s} doublet={rec.get('doublet_ok')!s:5s} "
          f"@{rec.get('h_ft')}/{rec.get('vc_kts')} eng={rec.get('n_engines')} | {rec.get('trim_msg','')[:70]} | {rec.get('doublet_msg','')[:40]}", flush=True)

df = pd.DataFrame(recs)
os.makedirs(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results")), exist_ok=True)
df.to_csv(os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "01_screen_tier_a.csv"), index=False)
print(f"\nPASS (trim+handsoff+doublet): {int(df.get('doublet_ok', pd.Series(dtype=bool)).fillna(False).sum())} of {len(df)}")

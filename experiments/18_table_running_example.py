"""Table 1 of the paper (running example): the Cessna 310 elevator doublet at selected time steps, baseline run
and candidate run with half the pitch damping, both with the same random draw (seed 100), no turbulence, the pitch rate, its level
and the level the dynamic signature predicts. Also prints the baseline violation rate of the pitch-rate rule
on the held-out correct runs. Writes paper/tables/tab_running.tex."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from devsig.run import load_run  # noqa: E402
from devsig.signature import Signatures  # noqa: E402

# the running example is flown without turbulence, as Figures 2 and 3, whatever DEVSIG_RESULTS says
RUNS = os.path.join(HERE, "..", "results_still", "runs", "tier_a", "c310")
TAB = os.path.join(HERE, "..", os.environ["DEVSIG_RESULTS"], "tables") if "DEVSIG_RESULTS" in os.environ else os.path.join(HERE, "..", "paper", "tables")
OUT = "velocities/q-rad_sec"
TIMES = [1.0, 2.0, 3.0, 3.5, 4.0, 8.0]
LEVELS = {5: ["very low", "low", "mid", "high", "very high"]}


def load(variant, seeds):
    paths = [os.path.join(RUNS, f"{variant}_elev_doublet_{s:04d}.parquet") for s in seeds]
    return [load_run(p) for p in paths if os.path.exists(p)]  # seeds whose trim failed are absent


def main():
    base = load("baseline", range(40))
    train, held = base[:len(base) // 2], base[len(base) // 2:]  # the split of the detection pipeline
    sig = Signatures(k=5, min_samples_leaf=10, lags_s=(0.5, 1.0, 2.0, 5.0), since_change=True).fit(train)
    vsd_held = float(np.mean([sig.vsd(r)[OUT] for r in held]))
    runs = {"baseline": load("baseline", [100])[0], "candidate": load("cmq_x0.5", [100])[0]}  # same random draw
    cols = {}
    for name, r in runs.items():
        p = sig._prep(r)
        d = sig.disc.transform(p[sig.features + sig.outputs])
        pred = sig.trees[OUT].predict(d[sig.features].to_numpy())
        t = r["simulation/sim-time-sec"].to_numpy(); t = t - t[0]  # time from the start of the manoeuvre (the trim precedes it)
        rows = []
        for tt in TIMES:
            i = int(np.argmin(np.abs(t - tt)))
            rows.append(dict(elev=float(r["fcs/elevator-cmd-norm"].iloc[i]), since=float(p["fcs/elevator-cmd-norm@since"].iloc[i]),
                             q=float(np.degrees(r[OUT].iloc[i])), level=LEVELS[5][int(d[OUT].iloc[i])], rule=LEVELS[5][int(pred[i])]))
        cols[name] = rows
    lines = ["\\begin{tabular}{r rrrll rrrll}", "\\toprule",
             "& \\multicolumn{5}{c}{baseline} & \\multicolumn{5}{c}{candidate} \\\\", "\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}",
             "$t$ (s) & elevator & since (s) & $q$ (deg/s) & level & rule & elevator & since (s) & $q$ (deg/s) & level & rule \\\\", "\\midrule"]
    n_dis = {k: 0 for k in runs}
    for j, tt in enumerate(TIMES):
        cells = [f"{tt:.1f}"]
        for name in runs:
            c = cols[name][j]
            mark = "" if c["level"] == c["rule"] else "$^{*}$"
            n_dis[name] += c["level"] != c["rule"]
            cells += [f"${c['elev']:+.2f}$", f"{c['since']:.1f}", f"${c['q']:.2f}$", c["level"], c["rule"] + mark]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, "tab_running.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nbaseline violation rate of the {OUT} rule on held-out runs: {vsd_held:.2f}; disagreements shown: {n_dis}")


if __name__ == "__main__":
    main()

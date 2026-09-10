"""Reach stage (protocol Section 5): whether a test case executes a line changed by a C++ real fault.
Each aircraft and test case is run once (seed 100, no turbulence) on a coverage-instrumented v1.3.1 build, in its own
process; llvm-cov gives the execution count of every line of the files the faults touch. The changed lines of a fault
are the v1.3.1 lines that its reversal edits (from `git diff -U0` in the fault's worktree); when the reversal only
inserts lines, the insertion point is used. Reach holds when any changed executable line runs at least once.
Usage: python experiments/25_coverage_reach.py <coverage build dir> <scratch dir with the worktrees>
Writes results/25_coverage_reach.csv."""
import json
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
t23 = __import__("23_real_fault_scores")
DEV_WT = {"B_pqrdot_5ad2694c": "jsbsim-r-pqrdot", "B_piston_power_bcd3f980": "jsbsim-r-piston", "B_turbine_windmill_2db8e408": "jsbsim-r-turbine"}

RUNNER = r'''
import sys, os, json
sys.path.insert(0, os.path.join(sys.argv[1], "src")); sys.path.insert(0, os.path.join(sys.argv[1], "experiments"))
ac, man = sys.argv[2], sys.argv[3]
root = os.path.join(sys.argv[1], "results", "variants", "B_lib_root")
if man == "low_pass":
    m = __import__("20_coverage_low_pass"); print(m.one(("cov_probe", 100, root)))
else:
    from devsig.manoeuvres import tier_a_run
    cfg = json.load(open(os.path.join(sys.argv[1], "config", "tier_a_aircraft.json")))["tier_a"][ac]
    df, msg, _ = tier_a_run(ac, cfg, man, 100, root_dir=root); print(msg)
'''


def changed_lines(wt):
    """file (relative) -> set of v1.3.1 line numbers edited by the reversal in worktree wt."""
    out = {}
    diff = subprocess.run(["git", "-C", wt, "diff", "-U0", "--", "src"], capture_output=True, text=True).stdout
    cur = None
    for line in diff.splitlines():
        if line.startswith("--- a/"):
            cur = line[6:]; out.setdefault(cur, set())
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+", line)
        if m and cur:
            a, b = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            out[cur] |= set(range(a, a + b)) if b > 0 else {a, a + 1}
    return out


def run_one(args):
    ac, man, cov_build, prof_dir = args
    prof = os.path.join(prof_dir, f"{ac}__{man}.profraw")
    if os.path.exists(prof):
        return ac, man, "cached"
    env = dict(os.environ, PYTHONPATH=os.path.join(cov_build, "tests"), LLVM_PROFILE_FILE=prof, DEVSIG_TURB_SEVERITY="0",
               DEVSIG_RESULTS="results_cov", OMP_NUM_THREADS="1")
    r = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"), "-c", RUNNER, REPO, ac, man], env=env, capture_output=True, text=True, cwd=REPO)
    return ac, man, (r.stdout.strip().splitlines() or ["?"])[-1][:60]


def counts(args):
    prof, so, files = args
    pd_ = prof.replace(".profraw", ".profdata")
    subprocess.run(["xcrun", "llvm-profdata", "merge", "-sparse", prof, "-o", pd_], check=True)
    lcov = subprocess.run(["xcrun", "llvm-cov", "export", "-format=lcov", f"-instr-profile={pd_}", so, *files], capture_output=True, text=True).stdout
    out, cur = {}, None
    for line in lcov.splitlines():
        if line.startswith("SF:"):
            cur = line[3:]; out[cur] = {}
        elif line.startswith("DA:") and cur:
            ln, c = line[3:].split(",")[:2]; out[cur][int(ln)] = int(c)
    return os.path.basename(prof)[:-8], out


def main():
    cov_build, scratch = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    src_root = os.path.dirname(cov_build)
    so = [os.path.join(cov_build, "tests", "jsbsim", f) for f in os.listdir(os.path.join(cov_build, "tests", "jsbsim")) if f.startswith("_jsbsim") and f.endswith(".so")][0]
    faults = {k: v for k, v in t23.FAULTS.items() if v[1] == "cpp"}
    wts = {k: os.path.join(scratch, DEV_WT[k]) if k in DEV_WT else os.path.join(scratch, "jsbsim-h-" + k.rsplit("_", 1)[1]) for k in faults}
    lines = {k: changed_lines(w) for k, w in wts.items()}
    files = sorted({os.path.join(src_root, f) for d in lines.values() for f in d})
    prof_dir = os.path.join(scratch, "coverage_profiles"); os.makedirs(prof_dir, exist_ok=True)
    battery = list(__import__("devsig.manoeuvres", fromlist=["BATTERY"]).BATTERY)
    jobs = [(ac, man, cov_build, prof_dir) for ac in t23.ALL for man in battery] + [("737", "low_pass", cov_build, prof_dir)]
    with Pool(int(os.environ.get("WORKERS", "3"))) as pool:
        for ac, man, msg in pool.imap_unordered(run_one, jobs):
            print(ac, man, msg, flush=True)
    profs = [os.path.join(prof_dir, f"{ac}__{man}.profraw") for ac, man, _, _ in jobs]
    with Pool(int(os.environ.get("WORKERS", "3"))) as pool:
        cnt = dict(pool.map(counts, [(p, so, files) for p in profs if os.path.exists(p)]))
    rows = []
    for key, per_file in cnt.items():
        ac, man = key.split("__")
        for k, d in lines.items():
            execd, total, mx = 0, 0, 0
            for f, ls in d.items():
                c = per_file.get(os.path.join(src_root, f), {})
                ex = [c[l] for l in ls if l in c]
                total += len(ex); execd += sum(1 for x in ex if x > 0); mx = max([mx] + ex)
            rows.append(dict(fault=k, aircraft=ac, test_case=man, changed_exec_lines=total, executed=execd, max_count=mx, reach=execd > 0))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(REPO, "results", "25_coverage_reach.csv"), index=False)
    print(out.groupby("fault").agg(lines=("changed_exec_lines", "max"), reach=("reach", "mean")).round(2).to_string())
    print(json.dumps({k: {f: sorted(v) for f, v in d.items()} for k, d in lines.items()}, indent=0)[:3000])


if __name__ == "__main__":
    main()

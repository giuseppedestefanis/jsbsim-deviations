"""Family B run generation. Three modes:
  xml   : reversed model-file fixes on the current models (normal venv): variants from mutate.FAMILY_B_XML
  lib   : a pre-fix JSBSim library (run with the alternative venv + PYTHONPATH to its module), 1.3.1 model files
          through a mirror root; the variant label is given on the command line.
  patch : a model-file fix reversed from config/real_fault_patches/<sha>.patch in a mirror root per aircraft
Usage:
  python experiments/12_family_b_runs.py xml <aircraft...>
  python experiments/12_family_b_runs.py patch <label> <sha> <aircraft...>
  PYTHONPATH=<old build tests dir> <old venv python> experiments/12_family_b_runs.py lib <label> <aircraft...>
Runs are written next to the Tier A runs: results/runs/tier_a/<ac>/<variant>_<manoeuvre>_<seed>.parquet
"""
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from devsig import mutate  # noqa: E402
from devsig.manoeuvres import BATTERY, load_tier_a_config, tier_a_run  # noqa: E402

RUNS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "runs", "tier_a")
VARIANTS = os.path.join(HERE, "..", os.environ.get("DEVSIG_RESULTS", "results"), "variants")
SEEDS = list(range(100, 110))


def one(args):
    aircraft, cfg, variant, manoeuvre, seed, root = args
    path = os.path.join(RUNS, aircraft, f"{variant}_{manoeuvre}_{seed:04d}.parquet")
    if os.path.exists(path):
        return variant, manoeuvre, "cached"
    df, msg, _ = tier_a_run(aircraft, cfg, manoeuvre, seed, root_dir=root)
    if df is None:
        return variant, manoeuvre, msg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    return variant, manoeuvre, "ok"


RUNNER = r"""
import sys, os, json
sys.path.insert(0, os.path.join(sys.argv[1], "src"))
from devsig.manoeuvres import tier_a_run
ac, man, seed, root, path = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
cfg = json.load(open(os.path.join(sys.argv[1], "config", "tier_a_aircraft.json")))["tier_a"][ac]
df, msg, _ = tier_a_run(ac, cfg, man, seed, root_dir=root)
if df is not None:
    os.makedirs(os.path.dirname(path), exist_ok=True); df.to_parquet(path)
print("RESULT:" + (msg if df is None else "ok"))
"""


def one_isolated(args):
    """Run one job in its own Python process: a faulty model that aborts the simulator (an uncaught C++ exception)
    kills only that process, and the run is recorded as failed (a structural regression)."""
    import subprocess
    aircraft, cfg, variant, manoeuvre, seed, root = args
    path = os.path.join(RUNS, aircraft, f"{variant}_{manoeuvre}_{seed:04d}.parquet")
    if os.path.exists(path):
        return variant, manoeuvre, "cached"
    try:
        r = subprocess.run([sys.executable, "-c", RUNNER, os.path.join(HERE, ".."), aircraft, manoeuvre, str(seed), root, path],
                           capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return variant, manoeuvre, "timeout"
    res = [l for l in r.stdout.splitlines() if l.startswith("RESULT:")]
    if res:
        return variant, manoeuvre, res[-1][7:]
    err = (r.stderr.strip().splitlines() or ["?"])[-1][:120]
    return variant, manoeuvre, f"simulator aborted (exit {r.returncode}): {err}"


def main():
    mode = sys.argv[1]
    cfgs = load_tier_a_config()["tier_a"]
    if mode == "xml":
        aircraft = sys.argv[2:]
        jobs = []
        for ac in aircraft:
            for name, factory in mutate.FAMILY_B_XML.items():
                try:
                    root = mutate.make_variant(ac, f"{ac}_{name}", VARIANTS, factory())
                except RuntimeError as e:
                    print(ac, name, "skipped:", e); continue
                jobs += [(ac, cfgs[ac], name, man, s, root) for man in BATTERY for s in SEEDS]
    elif mode == "lib":
        label, aircraft = sys.argv[2], sys.argv[3:]
        root = os.path.abspath(os.path.join(VARIANTS, "B_lib_root"))
        if not os.path.isdir(root):  # mirror of the installed 1.3.1 data: the reverted library reads the stock model files
            import sysconfig
            data = os.path.join(sysconfig.get_paths()["purelib"], "jsbsim")  # the pip package data, not the reverted build
            os.makedirs(root)
            for d in ("aircraft", "engine", "systems", "scripts"):
                os.symlink(os.path.join(data, d), os.path.join(root, d))
        jobs = [(ac, cfgs[ac], label, man, s, root) for ac in aircraft for man in BATTERY for s in SEEDS]
    elif mode == "patch":
        label, sha, aircraft = sys.argv[2], sys.argv[3], sys.argv[4:]
        subst = None
        if aircraft and aircraft[0] == "--subst":  # OLD=NEW applied to the aircraft file after the reversal
            subst, aircraft = aircraft[1].split("=", 1), aircraft[2:]
        patch = os.path.join(HERE, "..", "config", "real_fault_patches", f"{sha}.patch")
        jobs = []
        for ac in aircraft:
            try:
                root = mutate.make_reverted_root(ac, f"{ac}_{label}", VARIANTS, patch)
                if subst:
                    main = os.path.join(root, "aircraft", ac, f"{ac}.xml")
                    txt = open(main).read(); n = txt.count(subst[0])
                    open(main, "w").write(txt.replace(subst[0], subst[1])); print(ac, label, f"substituted {n} x {subst[0]!r} -> {subst[1]!r}")
            except RuntimeError as e:
                print(ac, label, "skipped:", e); continue
            jobs += [(ac, cfgs[ac], label, man, s, root) for man in BATTERY for s in SEEDS]
    else:
        raise SystemExit("mode must be xml, lib or patch")
    aborted = []
    if mode == "patch":
        # probe each aircraft once (cruise, seed 100); a model that aborts the simulator there is recorded as a
        # structural regression for all its runs without running them (each abort costs a crash report)
        keep = []
        for ac in sorted({j[0] for j in jobs}):
            aj = [j for j in jobs if j[0] == ac]
            probe = next((j for j in aj if j[3] == "cruise" and j[4] == 100), aj[0])
            _, _, msg = one_isolated(probe)
            if msg.startswith("simulator aborted"):
                print(ac, "probe aborted the simulator; all runs recorded as failed:", msg, flush=True)
                aborted += [(j[2], j[3], msg) for j in aj]
                with open(os.path.join(RUNS, ac, f"{aj[0][2]}_ABORTED.txt"), "w") as fh:
                    fh.write(msg + "\n")
            else:
                keep += [j for j in aj if j is not probe]
        jobs = keep
    with Pool(int(os.environ.get("WORKERS", "5"))) as pool:
        res = pool.map(one_isolated if mode == "patch" else one, jobs, chunksize=1 if mode == "patch" else 4)
    res = list(res) + aborted
    import collections
    print(collections.Counter((v, m) for v, _, m in res if m not in ("ok", "cached")))
    print("ok:", sum(1 for r in res if r[2] == "ok"), "cached:", sum(1 for r in res if r[2] == "cached"), "of", len(res))


if __name__ == "__main__":
    main()

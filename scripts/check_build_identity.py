"""Compare the pip wheel of JSBSim 1.3.1 with a locally built, unmodified v1.3.1 library, bit for bit.

  python scripts/check_build_identity.py run <prefix>             # runs five test cases (seed 100), saves <prefix>_<ac>_<test>.parquet
  PYTHONPATH=<local build>/tests python scripts/check_build_identity.py run <prefix2>
  python scripts/check_build_identity.py compare <prefix> <prefix2>   # writes results/build_identity.json

A difference between matched candidate and baseline runs of a revert-only build can then be attributed to the
reverted fix alone.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
os.environ["OMP_NUM_THREADS"] = "1"
CASES = [("737", "elev_doublet"), ("c172p", "throttle_step"), ("f16", "aileron_step"), ("c310", "engine_cut"), ("A320", "cruise")]


def run(prefix):
    from devsig.manoeuvres import tier_a_run
    import jsbsim
    cfg = json.load(open("config/tier_a_aircraft.json"))["tier_a"]
    root = os.path.abspath("results/variants/B_lib_root")
    for ac, man in CASES:
        df, msg, _ = tier_a_run(ac, cfg[ac], man, 100, root_dir=root)
        df.to_parquet(f"{prefix}_{ac}_{man}.parquet")
    print("library:", jsbsim.__file__)


def compare(a, b):
    out = []
    for ac, man in CASES:
        x, y = pd.read_parquet(f"{a}_{ac}_{man}.parquet"), pd.read_parquet(f"{b}_{ac}_{man}.parquet")
        cols = [c for c in x.columns if c in y.columns and np.issubdtype(x[c].dtype, np.number)]
        same = len(x) == len(y) and list(x.columns) == list(y.columns) and np.array_equal(x[cols].to_numpy(), y[cols].to_numpy())
        out.append(dict(aircraft=ac, test_case=man, rows=len(x), columns=len(cols), bit_identical=bool(same)))
        print(ac, man, "bit-identical" if same else "DIFFERENT")
    os.makedirs("results", exist_ok=True)
    json.dump(dict(cases=out, all_identical=all(r["bit_identical"] for r in out)), open("results/build_identity.json", "w"), indent=1)


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()

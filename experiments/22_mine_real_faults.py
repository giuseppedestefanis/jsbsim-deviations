"""Mine candidate real faults from the JSBSim history under Section 1 of protocol/real_fault_protocol.md.
Usage: python experiments/22_mine_real_faults.py <jsbsim clone> <worktree at v1.3.1>
Writes config/real_fault_candidates.csv: every commit that passes the automatic rules, with its kind (cpp/model),
files, and whether the inverse of the fix applies cleanly to v1.3.1. Classification (Section 2) is recorded by hand in
config/real_faults_heldout.csv."""
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MSG_IN = re.compile(r"\b(fix|fixed|fixes|bug|wrong|error|incorrect|mistake|sign)\b", re.I)
MSG_OUT = re.compile(r"typo|warning|compil|doc|comment|build|cmake|msvc|python|leak|segfault|crash|exception|print|output|locale|"
                     r"namespace|security|codeql|indent|assert|pars|pointer|header|const|unused|refactor|style|format", re.I)
CPP = ("src/models/", "src/math/", "src/initialization/")
MODEL = ("aircraft/", "engine/", "systems/")
IGNORE = ("tests/", "check_cases/", "doc/")
DEVELOPMENT = {"8819410c", "9c058118", "5ad2694c", "bcd3f980", "2db8e408"}


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, errors="replace").stdout


def main():
    repo, wt = sys.argv[1], sys.argv[2]
    log = git(repo, "log", "--no-merges", "--format=%x01%h%x02%ad%x02%s", "--date=short", "--name-only", "v1.3.1")
    rows = []
    for block in log.split("\x01")[1:]:
        head, *files = block.strip().split("\n")
        sha, date, msg = head.split("\x02")
        files = [f for f in files if f and not f.startswith(IGNORE)]
        if not files or not MSG_IN.search(msg) or MSG_OUT.search(msg):
            continue
        if all(f.startswith(CPP) for f in files):
            kind = "cpp"
        elif all(f.startswith(MODEL) for f in files):
            kind = "model"
        else:
            continue
        patch = git(repo, "show", "--format=", sha, "--", *files)
        chk = subprocess.run(["git", "-C", wt, "apply", "-R", "--check", "-"], input=patch, capture_output=True, text=True, errors="replace")
        rows.append(dict(sha=sha, date=date, kind=kind, message=msg, files=";".join(files), n_lines=patch.count("\n+") + patch.count("\n-"),
                         reverts_cleanly=chk.returncode == 0, development_set=sha[:8] in DEVELOPMENT))
    out = os.path.join(HERE, "..", "config", "real_fault_candidates.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    clean = [r for r in rows if r["reverts_cleanly"]]
    print(f"passed message and path rules: {len(rows)} (cpp {sum(r['kind'] == 'cpp' for r in rows)}, model {sum(r['kind'] == 'model' for r in rows)}); "
          f"revert cleanly on v1.3.1: {len(clean)} (cpp {sum(r['kind'] == 'cpp' for r in clean)}, model {sum(r['kind'] == 'model' for r in clean)}); "
          f"development set among them: {sum(r['development_set'] for r in clean)}")


if __name__ == "__main__":
    main()

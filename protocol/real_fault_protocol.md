# Protocol for the held-out real faults and the fault-revelation stages

Fixed on 2026-09-12, before any held-out candidate fault was inspected or run. The five real faults already in the
study (8819410c, 9c058118, 5ad2694c, bcd3f980, 2db8e408) form the development set; every fault selected below forms
the held-out set. Nothing in the test suite, the detector, the decision or the scoring changes after this point.

## 1. Mining rule (automatic)
- Source: the JSBSim git history up to and including the v1.3.1 tag (3b25f25e). Fixes made after v1.3.1 are excluded,
  since they are not present in the version under test.
- Commit message matches `\b(fix|fixed|fixes|bug|wrong|error|incorrect|mistake|sign)\b` (case-insensitive).
- The commit changes only files under `src/models/`, `src/math/`, `src/initialization/` (C++ faults) or model files
  under `aircraft/`, `engine/`, `systems/` (model-file faults), ignoring files under `tests/`, `check_cases/`, `doc/`.
- Excluded by message: `typo|warning|compil|doc|comment|build|cmake|msvc|python|leak|segfault|crash|exception|print|
  output|locale|namespace|security|codeql|indent|assert|pars|pointer|header|const|unused|refactor|style|format`.
- The inverse of the fix applies cleanly to v1.3.1 (`git revert --no-commit` without conflict for C++ faults;
  `patch -R --dry-run` without rejects against the shipped 1.3.1 model files for model-file faults). A fix that no
  longer applies cleanly is excluded as not reconstructible; no conflict is resolved by hand for the held-out set.
- The reverted C++ tree builds with the recipe of `scripts/build_prefix_library.md`.

## 2. Classification rule (by reading the diff, recorded with a reason per candidate)
A candidate is kept when all of the following hold:
- **Behavioural**: under some reachable flight state the fix changes a quantity computed by the flight dynamics,
  propulsion, atmosphere, mass, ground-reaction or initialisation code or model tables. Changes to error handling,
  exceptions, logging, output formatting, memory management, property binding without a change of value, parsing
  robustness, performance or the API are not behavioural.
- **Used by the subject**: the changed component is used by at least one of the 25 Tier A aircraft (engine type,
  system, or the aircraft/engine/system file itself), judged from the aircraft configuration files.
- **Single change**: the revert touches only the lines of that fix.
The decision and a one-line reason for every candidate that passed Section 1 are recorded in
`config/real_faults_heldout.csv` before any run of a held-out fault.

## 3. Frozen test suite and runs
- The fourteen open-loop test cases of the study (nine QTG-style manoeuvres, the climbing turn, the doublet with
  distractors, the engine restart, the engine stop, and the Boeing 737 low pass), unchanged, in both benign-variation
  conditions (no turbulence; moderate turbulence). No test is designed for a held-out fault.
- C++ faults run on all 25 aircraft with a library built from v1.3.1 with the one fix reverted; model-file faults run
  on the aircraft whose files carry the changed element. Ten candidate runs per test case (seeds 100-109) against the
  existing baseline runs of the same seeds (matched) and the twenty held-out baseline runs.
- The unmodified local build of v1.3.1 is bit-identical to the pip wheel on the test cases compared
  (`scripts/check_build_identity.py`), so any difference between matched runs is caused by the reverted fix.

## 4. Ground truth per (fault, aircraft, test case)
- **Structural regression**: a candidate run is missing (did not initialise, complete, or stay valid) where the
  matched baseline run completed.
- **Observable**: at least one recorded output has an effect size above 1 (matched-run distance over the benign
  variation between consecutive baseline runs), over all 22 outputs; no equation-derived oracle is used for the
  held-out faults.
- **Revealed**: structural regression, or the operational decision (Benjamini-Hochberg q = 0.05 over all outputs and
  test cases of the candidate version, i.e. one fault on one aircraft) flags at least one observable output.

## 5. Revelation stages and the signals evaluated
For every (fault, aircraft, test case) the stages of the RIPR model (reachability, infection, propagation,
revealability; Li and Offutt, after Voas's PIE) are measured:
- **Reach**: a line changed by the fix is executed. C++: line coverage of an instrumented v1.3.1 build, one run per
  aircraft and test case (seed 100, no turbulence). Model-file faults: the changed element is loaded by the aircraft
  (functions and tables are evaluated every frame), so reach holds by construction.
- **Propagation to the flight state**: the matched candidate and baseline runs (seed 100) differ bitwise in any
  recorded output. (Infection that never affects the flight state is invisible to every output-based check and is not
  separated.)
- **Observability**: as in Section 4, from ten matched pairs.
- **Revelation**: as in Section 4.
Signals compared as predictors of revelation, each computable before the full check: (a) membership of the
QTG-style battery; (b) reach by line coverage of the changed code; (c) propagation in one matched pair; (d) the
largest output divergence in one matched pair divided by the benign spread estimated from baseline runs only,
greater than 1. Reported: precision and recall of each signal for "the test case reveals the fault", per condition,
on the held-out set, with the development set shown separately; and the runs each signal saves.

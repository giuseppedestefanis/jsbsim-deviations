# jsbsim-deviations

Replication package for the study *Detecting behavioural changes in flight dynamics model updates*.

The package contains everything needed to regenerate every run, table and figure of the study: the code of the check (a residual detector with a statistically controlled, ranked decision, and a secondary rule-based attribution layer), the manoeuvre and mission definitions, the fault injection operators and the reverted-fix builds, the selection protocol for the real faults with its provenance record, the scoring scripts, the configuration of every aircraft and test case, and the result files of both conditions of operating variation.

## What the study does

A flight dynamics model (FDM) is the component of a flight simulator that computes how an aircraft moves in response to its controls and the air. When a supplier ships a new version, an engineer needs to know which recorded quantities now behave differently from the qualified version, and under which input. The study evaluates a check that needs only recorded runs: repeated runs of the qualified version under operating variation, and runs of the new version through the same tests. It asks three questions: how well the check and five other detectors find and rank changed outputs, and how often they flag an unchanged version (RQ1); why real faults are revealed or missed, from the execution of the changed code to a change of the recorded flight, its size against the variation between correct runs and a flag (RQ2); and what rule-based signatures add (RQ3). Matched replay, which compares runs with identical inputs and disturbances, is evaluated as a further comparator.

1. **Detector.** A regression model is trained, on runs of the qualified version, to predict each output of the aircraft (altitude, attitude, rates, accelerations) from the pilot's inputs and their recent history. On the new version, a one-sided permutation test per output compares the prediction errors of its runs with those of held-out correct runs, false-discovery-rate control over all outputs and test cases of the version gives the operational decision, and every output is ranked for inspection. A candidate run that does not load, finish or stay valid where the matched correct run did is reported first as a structural regression.
2. **Attribution (secondary).** Decision-tree rules ("signatures") are learnt on the same correct runs from discretised inputs and input history. For each flagged output the rule the new version breaks names an input, its level and its timing. In the study these rules find fewer of the large changes than every other detector and name the driving input no better than a baseline that names the input that moved most; they are kept in the package as evidence. Inputs that settle at a different constant position are reported separately as trim shifts.

The evaluation uses JSBSim 1.3.1 and its shipped aircraft models: fourteen open-loop test cases (Tier A: nine qualification-style manoeuvres, a multi-input climbing turn and an elevator doublet with distractor inputs on the 25 aircraft, and three coverage-driven tests designed after inspecting development faults, an engine restart on multi-engine jets, an engine stop on piston aircraft and a low pass of the 737), closed-loop missions through the autopilot on four aircraft plus a helicopter flight test (Tier B), four injected faults, seventeen faults rebuilt from real JSBSim bug fixes as they were shipped (each isolated to one change; five studied first and twelve selected under a protocol fixed in advance), no-op edits as false-alarm controls, six detectors compared under one decision protocol, and matched replay.

Everything is run under two conditions of operating variation: **no turbulence** (`DEVSIG_TURB_SEVERITY=0`, results in `results_still/`; objective qualification tests are flown in this condition) and **moderate turbulence** (the default, results in `results/`). Every experiment script honours `DEVSIG_RESULTS=<dir>` (where runs, result files, tables and figures go) and `DEVSIG_TURB_SEVERITY` (MIL-F-8785C exceedance level, 0 = off).

Operating variation between runs: fuel 55-100 %, payload offset (sd 15 lb), steady wind 0-8 kt in a random direction, initial altitude and airspeed jitter, input timing and amplitude jitter, and MIL-F-8785C turbulence at the third probability-of-exceedance level switched on after the trim with a per-run gust seed. Every draw comes from an independent stream derived from the run seed, the aircraft and the test case, so a run is reproducible across processes and a baseline and candidate run with the same seed share their inputs, wind, load and gusts exactly. The per-output decision is a one-sided permutation test (5,000 permutations) of the candidate's ten run scores against the twenty held-out correct runs on the statistic median(candidate) minus mean(held-out), at p < 0.05; its false-alarm rate is 5 per cent when candidate and held-out runs are exchangeable, and the no-op edits measure 5.7 per cent without turbulence and 2.7 per cent in turbulence for the residual. The operational decision applies the Benjamini-Hochberg procedure at q = 0.05 to all p-values of a version; the per-output p-values are stored (`08_per_output.csv`), so that level and stricter ones are computed without rerunning.

### Script names and the paper

Some script names carry the research-question labels of an earlier draft. In the paper's numbering: `08_baselines_rq2.py` is the detector comparison of RQ1; `09_explain_rq2b.py` is the attribution of RQ3; `13_family_b_score.py`, `14_windowed_rq4.py`, `16_tier_b_score.py`, `20_coverage_low_pass.py`, `23_real_fault_scores.py`, `24_ripr.py` and `25_coverage_reach.py` feed RQ2; `17_sensitivity_rq3.py` is the robustness appendix. Family A, B and C are the injected faults, the real faults and the no-op edits; Tier A and Tier B are the open-loop test cases and the closed-loop missions; the held-out set is the paper's protocol-selected set.

## Layout

| Path | Contents |
|---|---|
| `src/devsig/` | the library: `run.py` (run one JSBSim script, sample properties), `tier_a.py` (trim a model from Python, fly it, elevator doublet), `manoeuvres.py` (the nine-manoeuvre battery, benign variation, class-scaled input amplitudes), `missions.py` (closed-loop missions with phases, trim hand-over, autothrottle), `variation.py` (seeded benign variation: fuel, payload, wind, turbulence), `mutate.py` (fault injection by editing a copy of a model; mirror roots; axis-level operators; reversed real fixes), `discretize.py`, `signature.py` (static and dynamic signatures, VSD and RSD, windowed VSD), `baselines.py` (regression residual, envelope, KS test, isolation forest), `threshold.py` (null distribution and the permutation decision), `explain.py` (rule attribution, correctness, agreement), `evaluate.py`, `plots.py` |
| `experiments/` | one numbered script per step, listed below |
| `config/` | `tier_a_aircraft.json` (the 25 aircraft, their class, trim point and trim method, and the excluded models with reasons), `tier_b_aircraft.json` (autopilot aircraft, missions, helicopter phases), `real_fault_candidates.csv`, `real_faults_heldout.csv` and `real_fault_patches/` (the mined candidates, their classification and the stored fixes) |
| `protocol/` | the selection protocol for the real faults and `PROVENANCE.md` (creation times and hashes, and the two reconstruction changes made after the first runs) |
| `scripts/` | `regenerate_runs.sh` and `score_all.sh` / `score_rest.sh` (the full chain per condition), `rescore_all.sh` (rescore both conditions from stored runs), `heldout_runs.sh` (runs of the protocol-selected faults), `coverage_engine_stop.sh`, `run_rq2_all25.sh` (the six-detector comparison on all Tier A aircraft), `check_build_identity.py` (local build against the pip wheel), `build_prefix_library.md` (how to build a revert-only library) |
| `results/` | the result CSV files used for the tables and figures, and `headline_numbers.json`; generated runs go under `results/runs/` (not versioned, regenerated by the scripts) |
| `Dockerfile` | an image with the exact Python dependencies |

## Requirements and installation

Python 3.14, the version the study used (the Docker image pins it). JSBSim is installed from PyPI as a wheel that ships the aircraft, engine and system files; no separate JSBSim installation is needed. The library versions below are those of the study, as in `pyproject.toml` and the `Dockerfile`.

```bash
python3.14 -m venv .venv
.venv/bin/pip install jsbsim==1.3.1 pandas==3.0.5 numpy==2.5.3 scikit-learn==1.9.0 scipy==1.18.1 matplotlib==3.11.1 pyarrow==25.0.1
```

Or build the Docker image: `docker build -t jsbsim-deviations . && docker run --rm -it jsbsim-deviations`.

## Quick start (about ten minutes on one aircraft)

```bash
.venv/bin/python experiments/06_tier_a_battery.py c310      # Tier A runs and a first scoring for the Cessna 310
.venv/bin/python experiments/08_baselines_rq2.py c310       # six detectors on those runs
.venv/bin/python experiments/09_explain_rq2b.py c310        # attribution quality against the baselines
```

Each script prints a summary and writes a CSV under `results/`. Runs are cached as parquet files under `results/runs/`; a script only generates the runs that are missing, so it can be interrupted and restarted.

## Full reproduction, in order

| Step | Script | What it does | Output |
|---|---|---|---|
| 0 | `00_inventory.py` | runs every script shipped with JSBSim once | `results/00_inventory.csv` |
| 1 | `01_screen_tier_a.py` | trims every fixed-wing model and flies a hands-off minute and an elevator doublet; selects the Tier A aircraft | `results/01_screen_tier_a.csv`, `config/tier_a_aircraft.json` |
| 2 | `02_c310_pipeline.py`, `03_c310_diagnostics.py`, `04_c310_windowed.py`, `05_c310_graded.py` | the closed-loop circuit study on the Cessna 310: initial faults, effect sizes, windowing, graded magnitudes | `results/02_*` to `05_*` |
| 3 | `06_tier_a_battery.py <aircraft...>` | Tier A runs: thirteen manoeuvres, 50 baseline runs (40 for training and held-out, 10 sharing their draws with the candidates) and 10 runs per fault variant per manoeuvre, first scoring | `results/runs/tier_a/`, `results/06_*.csv` |
| 4 | `12_family_b_runs.py xml <aircraft...>` | real faults reproduced by reversing a model-file fix on the current models | `results/runs/tier_a/` |
| 4b | `12_family_b_runs.py lib <label> <aircraft...>` | real faults reproduced with a JSBSim 1.3.1 library in which one fix is reverted (see below), run with `PYTHONPATH` on that module | `results/runs/tier_a/` |
| 5 | `15_tier_b_runs.py` | closed-loop missions with phases | `results/runs/tier_b/` |
| 6 | `07_dynamic_signatures.py`, `08_baselines_rq2.py`, `09_explain_rq2b.py`, `17_sensitivity_rq3.py` (each `<aircraft...>`) | history features; six detectors with per-output records; attribution agreement, correctness, baselines and stability; robustness settings | `results/07_*.csv`, `08_baselines_rq2.csv`, `08_per_output.csv`, `09_*.csv`, `17_*.csv` |
| 7 | `13_family_b_score.py`, `14_windowed_rq4.py <aircraft...>`, `16_tier_b_score.py` | real faults with the effect-size gate (writes `paper/tables/tab_rq5.tex`); fixed windows on Tier A; Tier B whole-run, window, per-phase and max-over-phases scoring | `results/13_*.csv`, `14_*.csv`, `16_tier_b.csv` |
| 8 | `18_table_running_example.py`, `20_coverage_low_pass.py`, `19_analysis.py` | the running-example rule table (Cessna 310 elevator doublet, no turbulence, the matched seed-100 pair); the 737 low-pass coverage test; calibration, ranking measures, flag rate by effect size and hierarchical bootstrap intervals | `paper/tables/tab_running.tex`, `results/20_*.csv`, `results/19_analysis.json`, `paper/tables/tab_ci.tex` |
| 9 | `10_figures.py`, `11_results_25.py` | figures, tables (`tab_rq2`, `tab_rq2_gated`, `tab_rq2b`, `tab_rq4`) and `results/headline_numbers.json` | `paper/figures/`, `paper/tables/` (created if absent) |

`scripts/regenerate_runs.sh` runs steps 3, 4, 4b and 5 in order (set `JSBSIM_R_ROOT` to the directory holding the three reverted builds of the development faults; `scripts/heldout_runs.sh` uses the eight builds of the protocol-selected C++ faults, eleven in all); `scripts/score_all.sh` runs the scoring (steps 6 to 9, then the windowed Tier A and the robustness sweep last, via `scripts/score_rest.sh`); `scripts/coverage_engine_stop.sh` generates and scores the engine-stop coverage test; `experiments/21_conditions.py` lays the two conditions side by side (`paper/tables/tab_*_conditions.tex`, `results/conditions.json`). Run each chain once per condition, e.g. `DEVSIG_RESULTS=results_still DEVSIG_TURB_SEVERITY=0 scripts/regenerate_runs.sh`. The aircraft list is the key set of `config/tier_a_aircraft.json`. Generating the Tier A runs takes about an hour on twelve cores and scoring them about two hours; the scripts use one thread per worker (`OMP_NUM_THREADS=1`) and a process pool. Runs are cached, so an interrupted step resumes where it stopped.

## Faults

Injected faults (`mutate.py`) edit a copy of the aircraft XML and are applied through a mirror root, so every relative lookup resolves as in the stock installation. Operators: scale the moments of inertia and add a product of inertia; scale the first function in an aerodynamic axis whose name matches a pattern (pitch damping, pitch stiffness), which works across the naming conventions of the shipped models; add a lift offset as a new term in the lift axis; no-op rewrite (whitespace only) as a control.

Real faults come from the JSBSim history. Two are model-file fixes reversed on the current models whose files the fix changed: the drag term due to elevator deflection (commit 8819410c, 2020; ten Tier A models and the Tier B 737) and a kink in the ground-effect table of the 737 (9c058118, 2018). Three are C++ fixes: the angular-acceleration term (5ad2694c, 2024), negative piston power passed to the propeller when the engine is off (bcd3f980, 2020) and turbine windmilling from zero (2db8e408, 2021). For these, the fix is reverted on the 1.3.1 source, one fix per build, so that a candidate differs from the baseline by that change only (the piston revert also removes a later guard on the same quantity, which the pre-fix code did not have). Build each with its Python module (`scripts/build_prefix_library.md` has the full recipe) and run step 4b with `PYTHONPATH` pointing at the built module:

```bash
git clone https://github.com/JSBSim-Team/jsbsim.git jsbsim-src
git -C jsbsim-src worktree add ../jsbsim-r-pqrdot v1.3.1 && cd ../jsbsim-r-pqrdot && git revert --no-commit 5ad2694c
.venv/bin/pip install cython setuptools && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_MODULE=ON -DPython3_EXECUTABLE=<repo>/.venv/bin/python .. && make -j4
# the module is under build/tests; run: PYTHONPATH=<worktree>/build/tests .venv/bin/python experiments/12_family_b_runs.py lib B_pqrdot_5ad2694c <aircraft...>
```

On macOS with the command line tools add `-DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path) -DCMAKE_CXX_FLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"`. The reverted library runs the 1.3.1 model files through the mirror root `results/variants/B_lib_root` (four symlinks to the installed JSBSim data; the scripts create it). The variant labels are `B_pqrdot_5ad2694c`, `B_piston_power_bcd3f980` (piston aircraft) and `B_turbine_windmill_2db8e408` (jets).

## Protocol-selected (held-out) real faults and revelation stages

`protocol/real_fault_protocol.md` fixes, before any candidate was inspected, how further real faults are mined from the JSBSim history, which are kept, the frozen test suite and the quantities measured. The five real faults above are the development set; the protocol selects twelve more, the held-out set of the file names and the protocol-selected set of the paper. Of the 320 mined commits, 28 reverse cleanly on v1.3.1: twelve are kept, fourteen excluded with a recorded reason, and two are development faults.

| Step | Script | What it does | Output |
|---|---|---|---|
| H1 | `22_mine_real_faults.py <jsbsim clone> <worktree at v1.3.1>` | message and path rules, clean reversal on v1.3.1 | `config/real_fault_candidates.csv` |
| H2 | (by hand, recorded before any run) | classification with a reason per candidate | `config/real_faults_heldout.csv`, `config/real_fault_patches/<sha>.patch` |
| H3 | `scripts/heldout_runs.sh` (set `JSBSIM_H_ROOT`) | runs of every held-out fault on the fourteen test cases, one reverted build per C++ fault, a reversed patch in a mirror root per model-file fault, and the A-4 fault as the 2007 parser read it (`--subst`) | `results*/runs/` |
| H4 | `25_coverage_reach.py <coverage build> <dir with worktrees>` | one run per aircraft and test case on a coverage-instrumented v1.3.1 build; whether a line changed by each C++ fault executes | `results/25_coverage_reach.csv` |
| H5 | `23_real_fault_scores.py` (per condition; `ONLY_FAULTS=<label,...>` rescores listed faults and merges them) | structural failures, one-pair propagation, effect sizes and per-output p-values of all 17 real faults | `results*/23_real_fault_*.csv` |
| H6 | `24_ripr.py` | version-level decision; share of test cases passing each stage (reach, propagation, size beyond the benign variation, revelation counted strictly, as the protocol defines it, and broadly, a later analysis, with the chance level); precision and recall of the pre-check signals | `results/24_ripr.json`, `paper/tables/tab_ripr_*.tex` |

The literal reversal of the A-4 fault 3be71976 fails to load in 1.3.1, whose parser rejects the text `RETRACT`; the parser of 2007 read it as 0, wheels whose ground contact does not retract, and `B2_A4_gear_retractable_3be71976_as2007` reproduces that version and replaces the literal reversal in every count (`protocol/PROVENANCE.md` records when this was added).

The coverage build uses the recipe of `scripts/build_prefix_library.md` with `-fprofile-instr-generate -fcoverage-mapping` added to the compiler and linker flags. `scripts/check_build_identity.py` runs five test cases with the pip wheel and with an unmodified local build and compares them bit for bit; the result, all five identical, is in `results/build_identity.json`, so a difference between matched runs is caused by the reverted fix.

## Tier B notes

Only four shipped models have an autopilot (Cessna 172, Cessna 310, Global 5000, AH-1S). The shipped Global 5000 leaves its roll autopilot output unconnected; the study connects it and widens the altitude-error limit of the altitude hold from 100 to 500 feet in a variant of the model. The Boeing 737 flies with the Global 5000 autopilot attached in a variant. No shipped model connects its autopilot to the throttle, so the mission script holds airspeed with a proportional-integral autothrottle. `15_tier_b_runs.py` builds these variants under `results/variants/`.

## Result files

Each result CSV has one row per aircraft, manoeuvre or mission, method and fault variant, with recall, precision (NaN when nothing is flagged) and AUC against the physics-derived (primary) oracle, the gated values where the effect-size gate applies, the number of flagged outputs and, for the no-op variant, the false alarm rate. `results/08_per_output.csv` has one row per output (score, threshold, flag, oracle membership, effect size) and feeds `19_analysis.py`. `results/headline_numbers.json` and `results/19_analysis.json` collect the figures quoted in the text.

## Licence

Code: MIT. The aircraft models and JSBSim itself are distributed under their own licence (LGPL) and are not included here; they are installed from PyPI.

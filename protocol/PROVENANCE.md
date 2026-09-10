# Provenance of the protocol-selected real faults

Local file-system creation times (macOS birth time) and SHA-256 hashes of the files that fix the selection, in the order they were written. None of these files was modified after creation. These are local times recorded by the author's machine. No independent timestamp or preregistration service was used.

| created (local time, BST) | file | SHA-256 |
|---|---|---|
| 2026-09-12 11:49:27 | `protocol/real_fault_protocol.md` | `19f709d83ce4e23b2616a84307a58b5697938ceadad1f1f350dc0f839f4c4911` |
| 2026-09-12 11:51:16 | `experiments/22_mine_real_faults.py` | `22269e46cae4fbe4154dfb521e32ebaf686c8165fe25ae4696d5db023643fdb9` |
| 2026-09-12 11:54:57 | `config/real_fault_candidates.csv` | `65606b4db1f4c69c02766ea015faf1d1c0fac2ab5b405c4522833b50f56970aa` |
| 2026-09-12 12:18:35 | `config/real_faults_heldout.csv` | `a82b4ed6cde2ba1244b7b85fc9343bdb74ae11f872cc0e1d97f6bd3999a6ce7f` |
| 2026-09-12 12:27:02 | first run of a protocol-selected fault | |

The fourteen test cases (`src/devsig/manoeuvres.py`, `experiments/20_coverage_low_pass.py`) were defined before the protocol. Three later changes are to the harness and apply to every fault alike: the low-pass script gained a mode that runs one fault variant, runs of model-file faults are executed in separate processes so that a simulator abort is recorded as a failed run, and the heading angle is unwrapped when runs are loaded.

Two changes made after the first runs of the protocol-selected faults concern how a fault is reconstructed, and both are reported in the paper:

- The literal reversal of the A-4 fix 3be71976 fails to load in JSBSim 1.3.1, whose parser rejects the text value `RETRACT` of the `retractable` element. The parser of 2007, when the fault was shipped, converted that text with `atof` to 0, wheels whose ground contact does not retract. From 2026-09-12 16:38:05 the A-4 is also run with the value 0 (`B2_A4_gear_retractable_3be71976_as2007`, see `scripts/heldout_runs.sh`); this variant replaces the literal reversal in the counts, and the literal reversal is reported apart. The other two faults that stop the simulator were checked against the JSBSim source of their time and fail as they did then: the Boeing 314 property names (75b9b1e8, 2006) end in an undefined-property abort, and the F-4N holdback property (5306a18b, 2011) raises a missing-property exception.
- The two model-file faults of the development set are run only on the aircraft whose files their fixes changed: 8819410c on ten Tier A models (737, A4, B747, F4N, F80C, MD11, OV10, T37, T38, f15) and on the 737 in Tier B, and 9c058118 on the 737.

One change concerns how revelation is counted, and was also made after the protocol-selected faults had been run and scored:

- The protocol counts a test case as revealing a fault on a structural regression or when the operational decision flags an output whose effect size exceeds one; the paper calls this the strict count and reports it first. On 2026-09-12 (about 16:45, after the first scoring of the protocol-selected faults), a broad count was added in response to a review: a structural regression or a flag on any output the fault changed between matched runs (`revealed_any_change` in `experiments/24_ripr.py`). It is a later analysis and is reported as such. Terms: the protocol's "propagation" (a bitwise change of the seed-100 pair) is the paper's "the recorded flight changes", and the protocol's "observable" (an effect size above one) is the paper's "beyond the benign variation".

#!/bin/zsh
# Rescore every result of the study from the stored runs, in both conditions (used after a change to the loaders or
# the scoring, e.g. the heading unwrapping). Waits for any running real-fault scoring (23) to finish first.
cd "$(dirname "$0")/.."
while pgrep -f 23_real_fault_scores >/dev/null; do sleep 60; done
touch results/regenerate.done results_still/regenerate.done
rm -f results/score.done results_still/score.done results/essentials.done results_still/essentials.done
WORKERS=5 ./scripts/score_all.sh > results/rescore.log 2>&1 &
DEVSIG_RESULTS=results_still DEVSIG_TURB_SEVERITY=0 WORKERS=5 ./scripts/score_all.sh > results_still/rescore.log 2>&1 &
wait
.venv/bin/python experiments/21_conditions.py > results/21_conditions.log 2>&1
.venv/bin/python experiments/24_ripr.py > results/24_ripr.log 2>&1
echo "RESCORE DONE" > results/rescore.done

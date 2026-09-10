#!/bin/zsh
# Generate and score the engine-stop coverage test (piston aircraft) in one condition; run after the main chains' 13.
cd "$(dirname "$0")/.."
R_OUT=${DEVSIG_RESULTS:-results}; export OMP_NUM_THREADS=1
PISTON=(Boeing314 Camel J3Cub Short_S23 c172p c172r c182 c310 pa28)
R=${JSBSIM_R_ROOT:?set JSBSIM_R_ROOT}
until grep -q "^exit" $R_OUT/13_family_b.log 2>/dev/null; do sleep 60; done
.venv/bin/python experiments/06_tier_a_battery.py $PISTON > $R_OUT/06_engine_stop.log 2>&1; echo "exit $?" >> $R_OUT/06_engine_stop.log
PYTHONPATH=$R/jsbsim-r-piston/build/tests .venv/bin/python experiments/12_family_b_runs.py lib B_piston_power_bcd3f980 $PISTON > $R_OUT/12_engine_stop.log 2>&1; echo "exit $?" >> $R_OUT/12_engine_stop.log
.venv/bin/python experiments/13_family_b_score.py > $R_OUT/13_family_b.log 2>&1; echo "exit $?" >> $R_OUT/13_family_b.log
echo "ENGINE STOP DONE" > $R_OUT/engine_stop.done

#!/bin/zsh
# Regenerate every run of the study in order: Tier A battery (06), Family B model-file faults (12 xml),
# Family B library faults (12 lib, one revert-only build of JSBSim 1.3.1 per fault; see scripts/build_prefix_library.md),
# Tier B missions (15). Logs go to results/. Set JSBSIM_R_ROOT to the directory holding the three builds.
cd "$(dirname "$0")/.."
R_OUT=${DEVSIG_RESULTS:-results}; mkdir -p $R_OUT
export OMP_NUM_THREADS=1
ALL=(737 787-8 A320 A4 B747 Boeing314 Camel F4N F80C J3Cub MD11 OV10 Short_S23 T37 T38 c172p c172r c182 c310 f15 f16 global5000 pa28 pc7 t6texan2)
PISTON=(Boeing314 Camel J3Cub Short_S23 c172p c172r c182 c310 pa28)
JETS=(737 787-8 A320 A4 B747 F4N F80C MD11 T37 T38 f15 f16 global5000)
R=${JSBSIM_R_ROOT:?set JSBSIM_R_ROOT}
.venv/bin/python experiments/06_tier_a_battery.py $ALL > $R_OUT/06_all25.log 2>&1; echo "06 exit $?" >> $R_OUT/06_all25.log
.venv/bin/python experiments/12_family_b_runs.py xml $ALL > $R_OUT/12_family_b_xml.log 2>&1; echo "exit $?" >> $R_OUT/12_family_b_xml.log
PYTHONPATH=$R/jsbsim-r-pqrdot/build/tests .venv/bin/python experiments/12_family_b_runs.py lib B_pqrdot_5ad2694c $ALL > $R_OUT/12_family_b_pqrdot.log 2>&1; echo "exit $?" >> $R_OUT/12_family_b_pqrdot.log
PYTHONPATH=$R/jsbsim-r-piston/build/tests .venv/bin/python experiments/12_family_b_runs.py lib B_piston_power_bcd3f980 $PISTON > $R_OUT/12_family_b_piston.log 2>&1; echo "exit $?" >> $R_OUT/12_family_b_piston.log
PYTHONPATH=$R/jsbsim-r-turbine/build/tests .venv/bin/python experiments/12_family_b_runs.py lib B_turbine_windmill_2db8e408 $JETS > $R_OUT/12_family_b_turbine.log 2>&1; echo "exit $?" >> $R_OUT/12_family_b_turbine.log
.venv/bin/python experiments/15_tier_b_runs.py > $R_OUT/15_tier_b.log 2>&1; echo "exit $?" >> $R_OUT/15_tier_b.log
echo "ALL DONE" > $R_OUT/regenerate.done

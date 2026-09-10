#!/bin/zsh
# Score everything after scripts/regenerate_runs.sh: second Tier A pass (adds the three later manoeuvres; cached runs
# are skipped), then 07 and 08 on all aircraft, then scripts/score_rest.sh (09, real faults, Tier B, tables, figures,
# robustness last). Honours DEVSIG_RESULTS (results directory) and DEVSIG_TURB_SEVERITY.
cd "$(dirname "$0")/.."
R_OUT=${DEVSIG_RESULTS:-results}; mkdir -p $R_OUT
export OMP_NUM_THREADS=1
ALL=(737 787-8 A320 A4 B747 Boeing314 Camel F4N F80C J3Cub MD11 OV10 Short_S23 T37 T38 c172p c172r c182 c310 f15 f16 global5000 pa28 pc7 t6texan2)
while [ ! -f $R_OUT/regenerate.done ]; do sleep 60; done
.venv/bin/python experiments/06_tier_a_battery.py $ALL > $R_OUT/06_all25_pass2.log 2>&1; echo "exit $?" >> $R_OUT/06_all25_pass2.log
for s in 07_dynamic_signatures 08_baselines_rq2; do
  .venv/bin/python experiments/$s.py $ALL > $R_OUT/${s%%_*}_all25.log 2>&1; echo "exit $?" >> $R_OUT/${s%%_*}_all25.log
done
exec ./scripts/score_rest.sh

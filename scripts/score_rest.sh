#!/bin/zsh
# Scoring steps after 07 and 08 (resume of scripts/score_all.sh); robustness (17) last because it is the slowest.
cd "$(dirname "$0")/.."
R_OUT=${DEVSIG_RESULTS:-results}; export OMP_NUM_THREADS=1
ALL=(737 787-8 A320 A4 B747 Boeing314 Camel F4N F80C J3Cub MD11 OV10 Short_S23 T37 T38 c172p c172r c182 c310 f15 f16 global5000 pa28 pc7 t6texan2)
[ -f $R_OUT/09_explain_rq2b.csv ] || { .venv/bin/python experiments/09_explain_rq2b.py $ALL > $R_OUT/09_all25.log 2>&1; echo "exit $?" >> $R_OUT/09_all25.log; }
.venv/bin/python experiments/13_family_b_score.py > $R_OUT/13_family_b.log 2>&1; echo "exit $?" >> $R_OUT/13_family_b.log
.venv/bin/python experiments/16_tier_b_score.py > $R_OUT/16_tier_b.log 2>&1; echo "exit $?" >> $R_OUT/16_tier_b.log
.venv/bin/python experiments/18_table_running_example.py > $R_OUT/18_table.log 2>&1; echo "exit $?" >> $R_OUT/18_table.log
.venv/bin/python experiments/20_coverage_low_pass.py > $R_OUT/20_coverage.log 2>&1; echo "exit $?" >> $R_OUT/20_coverage.log
.venv/bin/python experiments/19_analysis.py > $R_OUT/19_analysis.log 2>&1; echo "exit $?" >> $R_OUT/19_analysis.log
.venv/bin/python experiments/10_figures.py > $R_OUT/10_figures.log 2>&1; echo "exit $?" >> $R_OUT/10_figures.log
.venv/bin/python experiments/11_results_25.py > $R_OUT/11_results.log 2>&1; echo "exit $?" >> $R_OUT/11_results.log
echo "ESSENTIALS DONE" > $R_OUT/essentials.done
.venv/bin/python experiments/14_windowed_rq4.py $ALL > $R_OUT/14_windowed.log 2>&1; echo "exit $?" >> $R_OUT/14_windowed.log
.venv/bin/python experiments/17_sensitivity_rq3.py $ALL > $R_OUT/17_all25.log 2>&1; echo "exit $?" >> $R_OUT/17_all25.log
echo "SCORING DONE" > $R_OUT/score.done

#!/bin/zsh
# Generate the runs of the held-out real faults (config/real_faults_heldout.csv) in one condition.
# Set JSBSIM_H_ROOT to the directory holding the worktrees jsbsim-h-<sha> (v1.3.1 with one fix reversed, built).
cd "$(dirname "$0")/.."
R_OUT=${DEVSIG_RESULTS:-results}; export OMP_NUM_THREADS=1
H=${JSBSIM_H_ROOT:?set JSBSIM_H_ROOT}
ALL=(737 787-8 A320 A4 B747 Boeing314 Camel F4N F80C J3Cub MD11 OV10 Short_S23 T37 T38 c172p c172r c182 c310 f15 f16 global5000 pa28 pc7 t6texan2)
LIBROOT=results/variants/B_lib_root
.venv/bin/python - <<'PY' > $R_OUT/heldout_list.txt
import pandas as pd
d = pd.read_csv("config/real_faults_heldout.csv"); d = d[d.decision == "keep"]
for _, r in d.iterrows():
    print(r.kind, r.sha, r.label, r.aircraft.replace(";", " "))
PY
while read kind sha label acs; do
  if [ "$kind" = "cpp" ]; then
    PYTHONPATH=$H/jsbsim-h-$sha/build/tests .venv/bin/python experiments/12_family_b_runs.py lib $label $ALL > $R_OUT/12_h_$sha.log 2>&1; echo "exit $?" >> $R_OUT/12_h_$sha.log
    PYTHONPATH=$H/jsbsim-h-$sha/build/tests .venv/bin/python experiments/20_coverage_low_pass.py run $label $LIBROOT > $R_OUT/20_h_$sha.log 2>&1; echo "exit $?" >> $R_OUT/20_h_$sha.log
  else
    .venv/bin/python experiments/12_family_b_runs.py patch $label $sha ${=acs} > $R_OUT/12_h_$sha.log 2>&1; echo "exit $?" >> $R_OUT/12_h_$sha.log
  fi
done < $R_OUT/heldout_list.txt
# the A-4 fault as the 2007 parser read it: atof converted the text RETRACT to 0, a gear that does not retract; the
# current parser rejects the text, so the literal reversal fails to load (added after the first runs, see PROVENANCE.md)
.venv/bin/python experiments/12_family_b_runs.py patch B2_A4_gear_retractable_3be71976_as2007 3be71976 --subst '<retractable>RETRACT</retractable>=<retractable>0</retractable>' A4 > $R_OUT/12_h_A4_2007.log 2>&1; echo "exit $?" >> $R_OUT/12_h_A4_2007.log
echo "HELDOUT RUNS DONE" > $R_OUT/heldout_runs.done

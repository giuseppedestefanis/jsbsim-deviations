#!/bin/zsh
# Six-detector comparison (RQ2) on all 25 Tier A aircraft, parallel over aircraft-manoeuvre pairs.
# Expected ~2 h on 12 cores. Results: results/08_baselines_rq2.csv, log: results/08_all25.log
cd "$(dirname "$0")/.."
.venv/bin/python experiments/08_baselines_rq2.py 737 787-8 A320 A4 B747 Boeing314 Camel F4N F80C J3Cub MD11 OV10 Short_S23 T37 T38 c172p c172r c182 c310 f15 f16 global5000 pa28 pc7 t6texan2 > results/08_all25.log 2>&1
echo "exit $?" >> results/08_all25.log

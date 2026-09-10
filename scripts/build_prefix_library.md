# Building a JSBSim 1.3.1 library with one fix reverted (Family B, C++ faults)

Each real fault is isolated to one change: the fix is reversed on the v1.3.1 source and the library is built with its
Python module, one build per C++ fault, eleven in all. The three development faults are `pqrdot` (5ad2694c,
`src/models/FGAccelerations.cpp`), `piston` (bcd3f980, `src/models/propulsion/FGPiston.cpp`) and `turbine`
(2db8e408, `src/models/propulsion/FGTurbine.cpp`), reverted with `git revert` as below; the eight protocol-selected
(held-out) faults are built from the stored fixes in `config/real_fault_patches/`, see the last section.

```bash
git clone https://github.com/JSBSim-Team/jsbsim.git jsbsim-src
for name_commit in pqrdot:5ad2694c piston:bcd3f980 turbine:2db8e408; do
  name=${name_commit%%:*}; commit=${name_commit##*:}
  git -C jsbsim-src worktree add ../jsbsim-r-$name v1.3.1
  git -C ../jsbsim-r-$name revert --no-commit $commit     # source only; resolve conflicts to the pre-fix logic
done
<repo>/.venv/bin/pip install cython setuptools
cd jsbsim-r-pqrdot && mkdir build && cd build
export SDKROOT=$(xcrun --show-sdk-path)                   # macOS with Command Line Tools
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_MODULE=ON \
      -DCMAKE_OSX_SYSROOT=$SDKROOT "-DCMAKE_CXX_FLAGS=-isystem $SDKROOT/usr/include/c++/v1" \
      -DPython3_EXECUTABLE=<repo>/.venv/bin/python -DPYTHON_EXECUTABLE=<repo>/.venv/bin/python -DCYTHON_EXECUTABLE=<repo>/.venv/bin/cython ..
make -j4        # about a minute per library; module under build/tests
```

Notes. The piston revert conflicts with a later guard on the same line (`if (RPM <= 0.1) power = max(power, 0.0)`);
resolve to the pre-fix line `Thruster->Calculate(HP * hptoftlbssec)`, which removes all negative-power filtering.
Check each build: `PYTHONPATH=<worktree>/build/tests <repo>/.venv/bin/python -c "import jsbsim; print(jsbsim.FGFDMExec().get_version())"`
prints 1.3.1. Then generate the runs on the 1.3.1 model files through the mirror root `results/variants/B_lib_root`:

```bash
PYTHONPATH=<worktree>/build/tests <repo>/.venv/bin/python experiments/12_family_b_runs.py lib B_pqrdot_5ad2694c <aircraft...>
```
or run all of them with `JSBSIM_R_ROOT=<dir holding the three worktrees> scripts/regenerate_runs.sh`.

## The eight protocol-selected (held-out) C++ faults

Their fixes are stored as patches and reversed with `git apply -R` on a v1.3.1 worktree; the mining script
(`experiments/22_mine_real_faults.py`) kept only fixes that reverse cleanly, so no conflict arises. Build each tree as
above and point `JSBSIM_H_ROOT` at the directory holding the worktrees before running `scripts/heldout_runs.sh`.

```bash
for sha in c8af9244 20066483 8a9fba1c f543d634 61f1e8c3 c3fa7d22 82cc3893 05b709c8; do
  git -C jsbsim-src worktree add ../jsbsim-h-$sha v1.3.1
  git -C ../jsbsim-h-$sha apply -R <repo>/config/real_fault_patches/$sha.patch
  (cd ../jsbsim-h-$sha && mkdir build && cd build && cmake <flags as above> .. && make -j4)
done
```

A tree is correct when the stored fix applies to it again cleanly (`git apply --check <patch>` succeeds).


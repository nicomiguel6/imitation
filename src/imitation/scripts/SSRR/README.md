## SSRR (Self-Supervised Reward Regression) — local implementation

This folder contains a **self-contained** implementation of SSRR inside the `imitation` codebase.

### What’s implemented here (per SSRR paper)
- **Phase 2 (Noise–Performance characterization)**: Fit a 4-parameter sigmoid
  \(\sigma(\eta)\) to AIRL-evaluated noisy rollouts (`curve_fit.py`).
- **Phase 3 (Reward regression)**: Train a reward network \(R_\theta(s,a)\) to
  regress snippet return \(\sum_t R_\theta(s_t,a_t)\) to \(\sigma(\eta)\) (with
  optional snippet-length scaling) (`reward_regression.py`).

### Double integrator first
The initial runner/tests target the existing double integrator environment:
`imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0`.

### Key modules
- `noise_rollouts.py`: collect noisy rollouts using the existing epsilon-greedy policy wrapper.
- `curve_fit.py`: compute AIRL-estimated returns per noise level and fit the sigmoid.
- `reward_regression.py`: snippet dataset + SSRR regression trainer.
- `runner_double_integrator.py`: end-to-end SSRR run for the double integrator (AIRL → noisy rollouts → fit curve → regress reward → RL).

### Notes / constraints
- This SSRR implementation does **not** modify the reference TensorFlow SSRR codebase under `/home/nicomiguel/SSRR`.
- We keep SSRR code isolated here until tests validate it; only then do we add an option into NTRIL as an alternative to D-REX.


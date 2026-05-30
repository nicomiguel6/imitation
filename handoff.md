# SSRR Regression Debug Handoff

## Session Goal

Diagnose why SSRR reward regression in `src/imitation/scripts/SSRR/tests/test_reward_regression_step.py` appears to "diverge" and produces a near-constant reward over state space.

## What Was Investigated

- Verified end-to-end regression logic in:
  - `src/imitation/scripts/SSRR/reward_regression.py`
  - `src/imitation/scripts/SSRR/tests/test_reward_regression_step.py`
  - `src/imitation/scripts/SSRR/curve_fit.py`
  - `src/imitation/scripts/SSRR/noise_rollouts.py`
- Checked whether training behavior matches intended SSRR objective:
  - batch of snippets
  - snippet cumulative reward prediction
  - MSE to sigmoid-derived target
- Computed dataset/target statistics from current AIRL run artifacts under:
  - `src/imitation/scripts/SSRR/tests/airl_outputs/20260527_231943_sinusoidal_A1.0_f0.01`

## Key Findings

1. **Training implementation matches intended loop**  
   The code does pass batches of variable-length snippets, computes cumulative predicted reward per snippet, compares to target return, and backprops via MSE.

2. **Observed behavior is not exploding divergence**  
   Loss is noisy but does not blow up numerically. It improves early, then plateaus.

3. **Primary failure mode = length shortcut / degenerate solution**  
   Learned predictions are almost perfectly explained by snippet length (constant per-step reward), not meaningful state discrimination.

4. **Removing weight decay alone is insufficient**  
   Degenerate behavior remains with `weight_decay=0`.

5. **AIRL-target comparison shows severe scale/sign mismatch**  
   AIRL snippet returns are on very different scale from sigmoid target regime, so direct comparison is dominated by mismatch.

## Code Changes Made

Added regression ablation diagnostics directly in:

- `src/imitation/scripts/SSRR/tests/test_reward_regression_step.py`

New helpers/functions:

- `_safe_corrcoef(...)`
- `_compute_airl_snippet_target(...)`
- `_evaluate_regression_model(...)`
- `run_regression_ablations(...)`

Main script now runs ablations before ensemble training via:

- `run_regression_diagnostics = True`

## Diagnostic Ablations Added

Each setting trains for 800 steps and reports:

- `final_loss`, `min_loss`
- `mse`
- `corr(pred,target)`
- `corr(pred,length)`
- `corr(target,length)`
- `pred_std`, `target_std`

Settings:

- `baseline_len_norm_wd`
- `fixed_len_wd`
- `baseline_no_wd`
- `baseline_airl_targets_no_wd`

## Reported Results (from terminal output)

- `baseline_len_norm_wd`: `corr(pred,length)=1.000`, `corr(pred,target)=0.547`
- `fixed_len_wd`: `pred_std=0.0009`, `corr(pred,target)=0.050` (length fixed, model nearly constant)
- `baseline_no_wd`: still `corr(pred,length)=1.000`
- `baseline_airl_targets_no_wd`: huge MSE, very large target std, negative correlation (scale/sign mismatch diagnostic)

## Interpretation

- The model is converging to a **length-driven constant-per-step reward** solution.
- This explains flat/blanket learned reward surfaces.
- Issue is **identifiability/objective structure**, not primarily vanishing gradients.

## Conceptual Direction Agreed

To avoid collapse, enforce signal that cannot be solved by length alone:

1. Remove length shortcut:
   - fixed snippet length, or
   - regress per-step mean reward rather than sum.
2. Add relative constraints:
   - ranking/pairwise loss between same-length snippets across noise levels.
3. Increase state-space diversity in rollouts.
4. Consider anti-collapse regularization and/or teacher-anchored local targets.
5. Align target scales before mixing AIRL-based and sigmoid-based objectives.

## Suggested Next Implementation Steps

1. Add CLI flags in `test_reward_regression_step.py`:
   - `--run_regression_diagnostics`
   - `--ablation_steps`
   - `--skip_rl`
2. Implement an alternative training mode:
   - fixed-length snippets only
   - loss on mean per-step return
3. Add ranking term for same-length snippet pairs:
   - enforce lower-noise snippet return > higher-noise snippet return by margin/logistic.
4. Add quick metrics after each run:
   - `corr(pred,length)`, `corr(pred,target)`, `pred_std`
   - fail-fast warning when collapse detected.

## Repo State Notes

There were pre-existing unrelated working tree changes and generated artifacts before this session (including SSRR AIRL output files and other modified scripts). This session intentionally did not revert them.

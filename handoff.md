# SSRR Reward Regression — Troubleshooting Handoff

## Goal

Get the SSRR learned reward (Phase 3 regression) to produce a well-shaped reward
surface near the reference trajectory, so the downstream RL agent actually
improves over the suboptimal expert. The pipeline lives in:

- `src/imitation/scripts/SSRR/reward_regression.py` — snippet dataset + `SSRRRegressor`
- `src/imitation/scripts/SSRR/curve_fit.py` — noise→return sigmoid fit (Phase 2)
- `src/imitation/scripts/SSRR/noise_rollouts.py` — noisy rollout generation (Phase 1)
- `src/imitation/scripts/SSRR/tests/test_reward_regression_step.py` — end-to-end driver
- `src/imitation/scripts/SSRR/util.py` — `plot_learned_reward_time_slider`
- `src/imitation/scripts/NTRIL/double_integrator/test_suboptimal_expert.py` — eval/compare

## Pipeline Recap (what trains on what)

1. **Phase 1 (noise rollouts):** `generate_noisy_rollout_buckets` rolls out the base
   policy at each noise level η. Noise metadata is stored *per-step* in
   `traj.infos[i]["noise_level"]` (and `noise_applied`, `total_applied_noise_sum`),
   not as a top-level trajectory attribute. Buckets pair η ↔ trajectories.
2. **Phase 2 (sigmoid fit):** `estimate_suboptimal_returns_by_noise` evaluates the
   reward net on each trajectory → `(η, mean_return)` points; `fit_sigmoid_noise_performance`
   fits the 4-param decreasing sigmoid σ(η).
3. **Phase 3 (regression):** `SnippetDataset` samples snippets, converts η→target via
   σ(η) (scaled by L/T when `length_normalize=True`), and `SSRRRegressor` trains
   R_θ(s,a) so the summed snippet reward matches that target (MSE).
4. **RL:** SAC trains on the ensemble-mean learned reward.

Key point: the network never sees η as an input — η only enters through the
**target label**. This is correct by design (η is unavailable at RL inference time).

## Changes Made This Session

1. **`curve_fit.py` / `types.py` — fixed flattened `returns_all`.**
   `NoisePerformanceData` gained `noise_levels_all` (paired 1:1 with flat `returns_all`).
   `estimate_suboptimal_returns_by_noise` now builds both in the same loop, so the
   scatter plot no longer relies on the caller re-iterating buckets in matching order.
   (`test_curve_fit_sigmoid.py` updated to use `noise_performance_data.noise_levels_all`.)

2. **`test_reward_regression_step.py` — fixed kwarg bug.**
   Two calls used `suboptimal_reward=reward_net`, but the function param is `reward`.
   Would `TypeError` only in the branch where `sigmoid_params.npy` is absent. Fixed both.

3. **`reward_regression.py` — batched forward pass.**
   `_collate_snippets` now flattens all snippet timesteps into single tensors;
   `SSRRRegressor.train` does ONE `reward_net(...)` call per batch, then `th.split` +
   sum per snippet (was a Python loop = one forward pass per snippet). Big speedup.

4. **`reward_regression.py` — dropped `next_obs`/`done` inputs.**
   Network input is now just `[obs; act]` (matches the original SSRR TF code).
   `BasicRewardNet` already defaults `use_next_state=False, use_done=False`.

5. **`util.py` — `plot_learned_reward_time_slider` zooms to reference range.**
   Was sweeping the full obs-space bounds (±100 pos / ±10 vel), so it plotted mostly
   extrapolation. Now zooms to reference trajectory range + `plot_margin` (default 0.5),
   clipped to obs bounds. **Plot-only fix; does not affect training.**

6. **`test_suboptimal_expert.py` — deterministic eval.**
   All learned-policy `predict()` calls (`ntril`, `drex`, `airl`, `ssrr`) now pass
   `deterministic=True`. This matters most for SSRR/SAC: stochastic sampling was
   injecting action noise every step and making the agent look like it never improved.

## Diagnosed Root Causes (reward regression quality)

- **State bounds leak into training data (real issue).**
  `DoubleIntegrator.step` *clamps* state to `±max_position` / `±max_velocity`
  (lines ~438–452 in `double_integrator.py`). With loose defaults (e.g. 100/10),
  high-noise rollouts drift far before clamping, so `RunningNorm` and the network
  are trained over a huge range while the reference lives in ≈[-1, 1]. Tighten bounds
  (e.g. `max_position≈3`, `max_velocity≈1` for A=1.0 sinusoid) to concentrate training
  data near the reference. NOTE: this changes the *data*, unlike the plot zoom in (5).

- **Snippet length / `target_scale` (objective structure).**
  Current driver uses `reg_min_steps = reg_max_steps = 1000` with `T = 1000`, so every
  snippet is the whole episode and `length_normalize` is a no-op — only 105 distinct
  training points (5 trajs × 21 noise levels). `target_scale` then acts as a pure global
  multiplier (why `1.0` is better-conditioned than `10.0` given small-init weights).
  - Short snippets (paper default) give far more diverse examples BUT the sinusoidal
    tracking reward is highly time-varying, so very short windows (10–50) bury the
    noise signal under positional variation.
  - Suggested middle ground: `reg_min_steps≈100, reg_max_steps≈200`, `target_scale≈5`,
    `reg_num_snippets≈20000`. Windows of ~150 cover >1 sinusoid period (period =
    1/f = 100 steps), averaging out reference variation while still giving ~90k distinct
    windows instead of 105.

- **Prior session finding (still relevant): length shortcut / degenerate solution.**
  With `length_normalize` + variable length, the model could explain predictions almost
  entirely by snippet length (`corr(pred,length)=1.0`), i.e. a constant per-step reward,
  rather than discriminating states. Mitigations discussed: fixed-length snippets,
  regress *mean* per-step return, add a ranking/pairwise term between same-length
  snippets across noise levels, and align scales before mixing AIRL vs sigmoid targets.

## Caching Gotchas (cause stale results)

- **Reward ensemble:** retrained only if `"reward_regression" in force_retrain` OR the
  `ensemble/ssrr_reward_*.pt` files are missing. After changing snippet/bounds/scale
  config, you MUST force retrain or you'll silently reuse old reward nets.
- **RL policy:** retrained only if `"rl_training" in force_retrain` OR
  `ssrr_rl/ssrr_rl_policy.zip` is missing. Same caveat.
- **Sigmoid params:** loaded from `sigmoid_fit/sigmoid_params.npy` if present.
- The latest policy is written to both `ssrr_rl/<timestamp>/ssrr_rl_policy.zip` and the
  top-level `ssrr_rl/ssrr_rl_policy.zip`. `test_suboptimal_expert.py` currently hardcodes
  a specific timestamped path — update it after each retrain (or point it at the
  top-level "latest").

## Suggested Next Steps

1. Re-run `test_suboptimal_expert.py` with the deterministic-eval fix to get a clean
   read on whether the *current* SAC policy is actually bad.
2. Tighten `max_position` / `max_velocity` in the env, force-retrain reward + RL, and
   re-check the (now zoomed) reward contour.
3. Sweep snippet length (≈100–200) and `target_scale`; watch `corr(pred,length)` vs
   `corr(pred,target)` to detect the length-shortcut collapse.
4. If SAC still underperforms on a good reward, add a PPO training path to A/B
   (deferred this session — was option (b), not yet done).

## Repo State Notes

Pre-existing unrelated working-tree changes and generated artifacts were present before
this session and were intentionally left as-is. No commits were made.

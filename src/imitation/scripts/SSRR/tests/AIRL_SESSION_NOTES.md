# AIRL Phase 1 — Session Notes

## Goal

Implement Phase 1 (AIRL) of the SSRR pipeline for the **DoubleIntegrator-v0** tracking task.
The SSRR paper (Chen, Paleja, Gombolay — CoRL 2020) uses AIRL only as a **bootstrapping step**
to produce a coarse reward function `R̃` and policy `π̃`. These are then used in SSRR Phase 2
(noise injection + sigmoid fitting + reward regression) and Phase 3 (SAC on the SSRR reward).

---

## Environment

- **Task**: Sinusoidal reference tracking with a double integrator (1D, unstable linear system)
- **Observation** (4D): `[position, velocity, ref_position, ref_velocity]`
- **Action** (1D): force, clipped to `[-20, 20]`
- **Episode length**: 1000 steps
- **Reward**: quadratic tracking error (per step) — highly nonlinear:
  - Expert (suboptimal K-gain policy): ~−501 total per episode
  - Random policy: ~−500,000,000 (exponential divergence once state leaves stable basin)
- **Key challenge**: The double integrator is an unstable system. There is a sharp boundary
  between the stable tracking region and the exponentially-diverging region. A stochastic policy
  occasionally crosses this boundary, causing catastrophic reward degradation.

**Suboptimal expert**: `DoubleIntegratorSuboptimalPolicy` with `K_position=0.02, K_velocity=0.3`.
This is a deterministic linear controller, essentially the "demonstration" source.

---

## SSRR Context

The SSRR paper does **not** use AIRL as the final reward. The pipeline is:

1. **Phase 1 — AIRL**: Run AIRL on suboptimal demos → produces `R̃` (initial reward) + `π̃` (base policy)
2. **Phase 2 — SSRR**: Inject noise at levels `η ∈ {0, 0.05, 0.10, 0.25, 0.50, 1.0}` into `π̃`,
   score noisy trajectories with `R̃`, fit a 4-parameter sigmoid `σ(η)` to the noise-performance
   curve, then regress a new reward `R_θ` via `L_SSRR = E[(Σ R_θ(s,a) - σ(η))²]`
3. **Phase 3 — SAC**: Train the final policy on `R_θ`

For Phase 1 to be useful for Phase 2, the base policy `π̃` (at η=0) must perform meaningfully
better than random — ideally approaching the suboptimal demonstration quality. The AIRL reward
only needs to correctly rank trajectories at different noise levels (~80%+ ranking accuracy is sufficient).

---

## Current Script

**Location**: `src/imitation/scripts/SSRR/tests/test_airl.py`

### Final Hyperparameters (current state of the script)

| Parameter | Value | Notes |
|---|---|---|
| `total_timesteps` | 2,000,000 | Best performance appears ~1.75M–1.875M steps |
| `n_envs` | 5 | Parallel envs via SubprocVecEnv |
| `n_steps` | 1000 | PPO rollout per env → 5000 steps/round |
| `batch_size` | 64 | PPO minibatch |
| `gamma` | 0.98 | Reduced from 0.99 — keeps value targets manageable |
| `gae_lambda` | 0.95 | |
| `ent_coef` | 0.01 | Reduced from 0.1 — lets policy converge toward deterministic expert |
| `learning_rate` | linear 3e-4 → 1e-5 | Prevents large updates late in training |
| `target_kl` | 0.02 | PPO early-stops update if KL exceeds this |
| `max_grad_norm` | 0.5 | |
| `net_arch` | [32, 32] | Policy and value network architecture |
| `demo_batch_size` | 512 | Discriminator batch size |
| `n_disc_updates_per_round` | 2 | Reduced from 10 → keeps reward signal stationary |
| `disc_lr` | 1e-3 | Discriminator Adam LR |
| `min_episodes` (demos) | 5 | Only 5 suboptimal demo episodes — potential weakness |

### Outputs per run

Each run creates `airl_outputs/<timestamp>_sinusoidal_A1.0_f0.01/` containing:
- `reward_net.pt` — final AIRL discriminator weights (post-training, may be post-blowup)
- `learner_policy.zip` — final PPO policy weights
- `best_checkpoint/reward_net.pt` — reward net at peak raw env reward ← **use this for SSRR**
- `best_checkpoint/learner_policy.zip` — policy at peak raw env reward ← **use this for SSRR**
- `results.npy` — before/after env reward statistics
- `config.json` — all hyperparameters
- `logs/` — TensorBoard + CSV logs (view with `tensorboard --logdir airl_outputs/`)

---

## Failure Modes Diagnosed and Fixed

### 1. Discriminator domination (original issue)
**Symptom**: `disc_acc` → 0.99 within first 20% of training; policy never learns.  
**Cause**: `n_disc_updates_per_round=10` → discriminator too dominant.  
**Fix**: Reduced to `n_disc_updates_per_round=2`.

### 2. Policy collapse (original issue)
**Symptom**: Policy `std` → 0.007; `approx_kl` spikes to 0.257; value_loss explodes.  
**Cause**: `ent_coef=0.01` insufficient, no KL guard, constant LR.  
**Fix**: `target_kl=0.02`, linear LR decay, `ent_coef=0.1` (later reduced to 0.01 for imitation).

### 3. Value function instability (persistent issue)
**Symptom**: `value_loss` oscillates between ~10k (normal) and 100k–1.5M (spikes).  
**Root cause**: With `gamma=0.99` and per-step AIRL reward ~−8, value targets ≈ −800.
Value MSE at initialisation ≈ 640,000. Discriminator updates abruptly shift the reward
function, causing transient target jumps.  
**Partial fix**: Reduced `gamma=0.98` (targets ≈ −400, MSE ≈ 160k) + `n_disc_updates_per_round=2`
(halves per-round reward shift).

### 4. End-of-training blowup (fundamental, partially mitigated)
**Symptom**: After reaching near-peak performance (~−365k raw env reward), a single
5k-step window sees reward jump to −12M. `value_loss` immediately spikes to 1.9M.
Policy `std` and `approx_kl` are both tiny at the moment of trigger — the policy did not change.  
**Root cause**: The DoubleIntegrator has a sharp stability basin. With `std≈0.155`,
the stochastic policy occasionally samples an action that tips the state into the
exponentially-diverging regime. Once outside the basin, the AIRL reward provides no gradient
back — it only says "you're far from the demonstrations". The value function, calibrated to
the stable regime, then faces catastrophically wrong targets.  
**Mitigation**: Best-checkpoint callback saves the reward net and policy at the peak moment,
so the blowup at training's end is irrelevant for downstream SSRR use.

### 5. Discriminator LR too low (introduced and reversed)
**Attempted fix**: Reduced `disc_lr=1e-3` → `3e-4` to slow discriminator reward shifts.  
**Result**: Discriminator entropy stayed at 0.23 after 1.3M steps (should be <0.05);
policy `std` barely moved (1.01 → 0.83 in 1.3M steps); policy never learned.  
**Conclusion**: `disc_lr=1e-3` is needed for fast enough discriminator convergence.
The stationary-reward problem is better addressed by `n_disc_updates_per_round`.

---

## Reward Ranking Quality (best checkpoint from run `20260420_222741`)

This is the best run so far — peak raw env reward ≈ −365k (vs expert −501, random −500M).

| Policy | `env_rew` | `airl_rew` |
|---|---|---|
| Expert | −461 | +3,362 |
| Learned η=0.00 | −345M (post-blowup) | −75,672 |
| Learned η=0.05 | −596M | −130,306 |
| Learned η=0.25 | −728M | −179,833 |
| Learned η=0.50 | −709M | −291,319 |
| Random η=1.00 | −575M | −481,978 |

**Ranking accuracy**: 80%+ (AIRL reward is monotonically decreasing with noise level).  
**Problem**: The saved checkpoint (end-of-training) is post-blowup. The policy at η=0
performs at −345M, not −365k. The best_checkpoint callback was not yet added to this run.

**The best available checkpoint for SSRR Phase 2** is therefore the best_checkpoint from
the most recent stable run, once the checkpoint callback has captured the −365k peak moment.

---

## What the SSRR Phase 2 Needs from Phase 1

1. **AIRL ranking accuracy ≥ 80%** — achieved
2. **Base policy (η=0) near-demonstration performance** — the peak of −365k is 720× worse
   than the expert (−501), which is workable. The post-blowup −345M is not workable.
3. **Reward net saved at peak, not at end of training** — the `best_checkpoint/` directory
   now handles this automatically

---

## Key Files

| Path | Description |
|---|---|
| `src/imitation/scripts/SSRR/tests/test_airl.py` | Main AIRL Phase 1 script |
| `src/imitation/algorithms/adversarial/airl.py` | AIRL class (imitation library) |
| `src/imitation/algorithms/adversarial/common.py` | AdversarialTrainer — `train(callback=)` |
| `src/imitation/scripts/NTRIL/double_integrator/double_integrator.py` | Env + suboptimal policy |
| `airl_outputs/20260420_222741_*/` | Best run so far (reached −365k peak) |
| `airl_outputs/20260420_222741_*/logs/raw/gen/progress.csv` | Generator metrics CSV |
| `airl_outputs/20260420_222741_*/logs/raw/disc/progress.csv` | Discriminator metrics CSV |

---

## Next Steps

1. **Run the current script** (with best_checkpoint callback) to obtain a reward net and
   policy saved at the actual peak performance moment (not end-of-training).
2. **Verify ranking** of the best_checkpoint using the evaluation script pattern above.
3. **Implement SSRR Phase 2**:
   - Load `best_checkpoint/learner_policy.zip` and `best_checkpoint/reward_net.pt`
   - Generate noisy trajectories at noise levels `η ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25}`
     (paper uses 10 trajectories per training iteration)
   - Score each trajectory with the AIRL reward net: `R̃(τ) = Σ_t reward_net(s_t, a_t, s_{t+1}, done_t)`
   - Fit a 4-parameter sigmoid `σ(η) = c / (1 + exp(-k(η - x₀))) + y₀` to `{η_i, R̃(τ_i)}`
   - Train `R_θ` (new MLP reward net) via SSRR loss:
     `L = E_τ [(Σ_t R_θ(s_t, a_t) - σ(η))²]` with L2 regularisation weight 0.1
   - Use snippets of length 50–500 steps (not full 1000-step trajectories) as in the paper
4. **Implement SSRR Phase 3**: Train SAC on `R_θ` (SAC handles continuous action spaces
   better than PPO for this task; the paper uses SAC for all final policies).

---

## Monitoring

```bash
# View all runs in TensorBoard
tensorboard --logdir src/imitation/scripts/SSRR/tests/airl_outputs/

# Quick CSV inspection
python3 -c "
import pandas as pd
gen = pd.read_csv('src/imitation/scripts/SSRR/tests/airl_outputs/<run>/logs/raw/gen/progress.csv')
print(gen[['raw/gen/rollout/ep_rew_mean','raw/gen/rollout/ep_rew_wrapped_mean',
           'raw/gen/train/std','raw/gen/train/value_loss',
           'raw/gen/train/approx_kl','raw/gen/time/total_timesteps']].iloc[::20].to_string())
"
```

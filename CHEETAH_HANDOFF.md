# CHEETAH_HANDOFF.md — SSRR/NTRIL on MuJoCo (HalfCheetah-first)

> **You are the agent picking up this work in a new chat.**
> **DO NOT create folders or write code yet.** First read this whole document,
> then **ask the user the clarifying questions in Section 8** and wait for
> answers. Only after the user responds should you scaffold anything. The user
> explicitly wants to confirm structure and reuse decisions before you build.

---

## 1. Background: what the previous chat accomplished (double integrator)

We debugged and (largely) fixed an SSRR pipeline on a 1-D double-integrator
tracking task. The pipeline lives in `src/imitation/scripts/SSRR/` with the
end-to-end driver `SSRR/tests/test_reward_regression_step.py`. Phases:

1. **AIRL** → base reward net + base (suboptimal) policy.
2. **Phase 1 (noise rollouts)** `noise_rollouts.generate_noisy_rollout_buckets`
   — roll out the base policy at noise levels η via epsilon-greedy injection.
3. **Phase 2 (sigmoid fit)** `curve_fit` — evaluate AIRL return per noise level,
   fit 4-param decreasing sigmoid σ(η).
4. **Phase 3 (reward regression)** `reward_regression.SSRRRegressor` — regress
   summed snippet reward Rθ(s,a) to σ(η) targets (MSE).
5. **RL** — SAC trains on the ensemble-mean learned reward.

### Root causes we diagnosed
- **State-distribution collapse from the noise process.** The original
  epsilon-greedy injector replaced actions with a **full-range** uniform random
  action. On a marginally-stable integrator, *any* η ≳ 0.05 instantly saturated
  the state to its clamp bounds, so ~20 of 21 noise buckets visited the **same
  boundary states** with **different** σ(η) targets → unsatisfiable regression →
  near-constant ("flat") reward. (Numbers: AIRL reward dynamic range ≈ 301 with a
  ridge through the reference; SSRR reward range ≈ 0.002 with its argmax stuck at
  the state-space boundary — i.e. correct-looking contour, but ~150,000× smaller
  range and pointing the wrong way.)
- **Flat + positive reward starves SAC.** Advantage ≈ 0 everywhere and SAC's
  entropy term dominates → near-random policy.

### Fixes we implemented (all reversible / opt-in; defaults preserve old behavior)
1. **`NTRIL/noise_injection.py`** — `EpsilonGreedyNoiseInjector(noise_action_scale=1.0)`.
   `scale < 1.0` shrinks the random replacement action toward the action
   midpoint → **graded** degradation. `1.0` reproduces original SSRR/D-REX. We
   verified scale `0.1` turns the saturating `|pos|` spectrum
   (`0.83→6.81→7.6→7.7→7.5`) into a smooth one (`0.81→2.75→5.43→7.66→8.94`).
2. **`SSRR/noise_rollouts.py`** — plumbs `noise_action_scale` through.
3. **`SSRR/util.py::reward_sanity_gate`** — pre-RL gate that checks the reward is
   oriented toward the reference: per-frame argmax distance, on-reference
   percentile, flatness (std/|mean|), and now also returns `reward_grid_mean` /
   `reward_grid_std`. **This gate is 2-D (pos,vel) and reference-tracking
   specific — it will NOT transfer to HalfCheetah and must be replaced (see §5).**
4. **Driver** — scale-specific artifact caching (`_scale0p1` tag) so switching
   `noise_action_scale` never clobbers original results; sanity gate wired in
   before RL (`block_rl_on_gate_fail`); and an **RL-side affine reward rescale**
   `r' = a·(r − b)` (`reward_output_scale="auto"`, `reward_output_center=True`,
   `reward_target_std`). A positive affine transform doesn't change the optimal
   policy but brings the tiny (~1e-3) reward to ~O(1) so it isn't drowned by SAC's
   entropy bonus.
5. Fixed a latent `SigmoidParams` subscripting bug exposed by the new cache tag.

### Outcome on the double integrator
With graded noise (`scale=0.1`), converged regression, and `target_std≈50`, SAC
produces a **non-degenerate, phase-locked** policy (learns the reference
*frequency*, ~100-step period) but with **loose amplitude/centering** — the
signature of a correctly-oriented but **broad/shallow** reward basin (a softened
version of AIRL's sharp ridge). We judged the **pipeline plumbing validated** and
decided the *undamped* integrator is an *adversarial* case for SSRR (see §6) —
which motivates both adding **damping** (Track A) and moving to **MuJoCo**
(Track B).

> Existing per-chat context: repo root `handoff.md` (earlier session notes) and
> `SSRR/tests/AIRL_SESSION_NOTES.md`. A reference TensorFlow SSRR codebase exists
> at `/home/nicomiguel/SSRR` (do not modify it; use only as a spec reference).

---

## 2. Goal for this phase

**Top priority: reproduce SSRR-paper results.** Stand up SSRR mirroring the
double-integrator architecture, **reusing the env-agnostic SSRR core**, organized
into a new `SSRR` (replication) subfolder and a `NTRIL` (advancement) subfolder.

Work is split into **two environment tracks** (see §3) so SSRR can be validated on
cheap, well-behaved linear systems *and* on the MuJoCo benchmarks the paper uses:

- **Track A — linear, dissipative testbeds:** damped double integrator and a
  linearized unicycle. These keep dynamics **linear**, so the *existing*
  `NTRIL/robust_tube_mpc.py` is **directly reusable**, and they're easier SSRR
  testbeds than the undamped integrator.
- **Track B — nonlinear MuJoCo locomotion:** HalfCheetah first (Hopper/Ant via
  config). SSRR replication is the goal here; **robust MPC is dropped for now**
  and left as placeholder boilerplate (§5, Q-list).

Milestones, in order:
1. **SSRR replication** on the chosen env(s) — suboptimal expert → noisy rollouts
   → sigmoid → reward regression → RL that improves over the expert.
2. **NTRIL advancement (linear MPC first):** layer the linear robust tube-MPC data
   augmentation on the Track-A linear envs. The nonlinear-MuJoCo MPC is explicitly
   deferred (placeholder only).

---

## 3. Environment tracks

### Track A — linear, dissipative testbeds (SSRR **and** linear tube MPC)

These are the recommended places to get SSRR + the existing linear robust tube
MPC working together, since dynamics stay linear and dissipative (graceful
degradation under noise — exactly the property the undamped integrator lacked).
They **retain reference tracking** (and the linear tube MPC tracks a reference).

1. **Damped double integrator (option to add).** Add a `damping` coefficient to
   the existing env (`NTRIL/double_integrator/double_integrator.py`): continuous
   `ẍ = u − c·ẋ`, i.e. discrete `A = [[1, dt],[0, 1 − c·dt]]`, `B = [[0],[dt]]`.
   - Default `damping = 0.0` preserves today's behavior (reversible).
   - `c > 0` makes `A` asymptotically stable → noisy rollouts degrade gracefully
     instead of saturating the clamp bounds → **much easier SSRR**, and the
     linear tube MPC still applies (just a different stable `A`).
   - Prefer adding the kwarg to the existing env (and threading it into the
     registered `DoubleIntegrator-v0` make-kwargs) over a new class, to maximize
     reuse of the existing AIRL/SSRR/MPC plumbing.
2. **Linearized unicycle (new env).** Unicycle `ẋ=v cosθ, ẏ=v sinθ, θ̇=ω`,
   controls `(v, ω)`. Linearize about a reference trajectory/operating point to
   get LTI (or LTV) `A, B` suitable for SSRR snippets and the linear tube MPC.
   Implement as a small Gymnasium env mirroring the double-integrator env's API
   (obs = state [+ ref], `Q`/`R` tracking reward, `A`/`B` exposed for the MPC).
   Confirm linearization scheme with the user (Q-list).

### Track B — nonlinear MuJoCo locomotion (SSRR replication only, for now)

HalfCheetah first; Hopper/Ant behind config. **Stationary** locomotion reward
(forward velocity − ctrl cost) → **no reference tracking** (drop the
`reference_trajectory` plumbing and `[state, ref]` obs augmentation; reward net
sees raw obs+act). Robust MPC is **not** implemented here yet — see §5.

---

## 4. Proposed folder structure (CONFIRM with user before creating)

A new top-level folder under `src/imitation/scripts/`, with SSRR (replication) and
NTRIL (advancement) subfolders. Naming TBD per Q1. Strawman:

```
src/imitation/scripts/ssrr_ntril/        # name CONFIRMED by user
├── __init__.py
├── README.md
├── common/                              # shared, env-agnostic glue
│   ├── envs.py                          # env registry/config: damped DI, unicycle, HalfCheetah/Hopper/Ant
│   ├── experts.py                       # train/load (sub)optimal experts (+ AIRL base reward)
│   ├── reward_eval.py                   # sanity gate: 2-D grid (linear envs) AND held-out return-corr (high-dim)
│   └── plotting.py                      # contours (linear) + noise→return / learned-vs-true return
├── SSRR/                                # milestone 1: replicate SSRR (all envs, config-driven)
│   ├── runner.py
│   └── tests/
└── NTRIL/                               # milestone 2: robust-MPC advancement
    ├── runner.py                        # linear tube MPC on Track-A envs
    ├── mpc_nonlinear_placeholder.py     # boilerplate stub for MuJoCo MPC (deferred)
    └── tests/
```

SSRR core modules (`curve_fit`, `noise_rollouts`, `reward_regression`, `types`,
`reporting`) and `NTRIL/robust_tube_mpc.py` stay where they are and are
**imported**, not copied.

---

## 5. Reuse map

| Component | File(s) | Reuse plan |
|---|---|---|
| Noise injection (graded) | `NTRIL/noise_injection.py` | **Reuse as-is.** Track A: tune `noise_action_scale` for graceful degradation (damping helps). Track B (MuJoCo, action ~[-1,1]): start `noise_action_scale=1.0`. |
| Noisy rollout buckets | `SSRR/noise_rollouts.py` | **Reuse as-is** (`reference_trajectory` is optional — keep for Track A, drop for Track B). |
| Sigmoid fit (Phase 2) | `SSRR/curve_fit.py` | **Reuse as-is.** |
| Reward regression (Phase 3) | `SSRR/reward_regression.py`, `SSRR/types.py` | **Reuse as-is.** Input `[obs; act]` already env-agnostic. Revisit `length_normalize`/`target_scale` for high-dim. |
| Run reporting | `SSRR/reporting.py` | Reuse; may add fields. |
| **Linear robust tube MPC** | `NTRIL/robust_tube_mpc.py` | **Reuse on Track A** (damped DI, linearized unicycle): linear `A,B`, `solve_discrete_are`, polytopes — already fits these. Only swap the double-integrator-specific `generate_reference_trajectory` import / `A,B` source per env. |
| NTRIL trainer | `NTRIL/ntril.py`, `demonstration_ranked_irl.py`, `ranked_dataset.py` | Reuse ranking/IRL/orchestration; pair with the linear MPC on Track A. |
| Nonlinear (MuJoCo) MPC | — | **Deferred.** Add `mpc_nonlinear_placeholder.py` boilerplate stub only; no real implementation this phase. |
| Reward sanity gate / plots | `SSRR/util.py` | **Reuse the 2-D `reward_sanity_gate` + contour plots for Track A** (DI/unicycle are low-dim). **Add** a high-dim variant (held-out learned-vs-true return correlation + noise-monotonicity) for Track B. |
| Driver / DI-hardcoded env bits | `SSRR/tests/test_reward_regression_step.py`, `NTRIL/double_integrator/*` | Generalize into the config-driven `runner.py`; reuse DI/MPC plumbing for Track A, don't hardcode it for Track B. |

---

## 6. Key conceptual differences by track

- **Track A (damped DI, unicycle):** linear + dissipative + reference-tracking.
  The existing 2-D `reward_sanity_gate`, contour plots, AIRL path, and **linear
  tube MPC** all transfer. Damping is the key lever that should make SSRR behave
  far better than the undamped integrator did (graceful degradation instead of
  boundary saturation).
- **Track B (MuJoCo):** nonlinear, high-dim, **stationary** locomotion reward.
  - Drop reference tracking / obs augmentation; reward net sees raw obs+act.
  - It's actually *easier* for SSRR than the undamped integrator: dissipative
    dynamics, noise averaging across many joints, naturally O(1–1000) rewards.
  - Sanity gate must be high-dim: **rank/Pearson correlation between learned-
    reward return and ground-truth env return** on held-out rollouts, plus
    **learned-return-vs-noise monotonicity**. Keep the "gate before RL" pattern.
  - Keep the affine `r'=a·(r−b)` reward rescale for SAC.
  - Robust MPC: **placeholder only** this phase.

---

## 7. Environment generalization — recommendation

**Generalize the SSRR + RL replication across envs; keep MPC per-track.**

- The SSRR core is already env-agnostic. Use a **config-driven runner** (env spec
  + expert spec + hyperparams in `common/envs.py`) covering damped DI, unicycle,
  HalfCheetah, Hopper, Ant. Do **not** fork per-env SSRR runners.
- MuJoCo locomotion envs share the Gymnasium API and reward structure (forward
  progress − ctrl cost), so Hopper/Ant cost little once HalfCheetah works (and the
  SSRR paper reports all three).
- **Tailor only the dynamics-coupled MPC**: linear tube MPC for Track A (reused),
  nonlinear MuJoCo MPC deferred behind a stub interface.
- Practical caution: validate **one env per track** end-to-end first (damped DI
  for Track A; HalfCheetah for Track B), then flip on the others via config.

---

## 8. Questions to ask the user (ASK THESE FIRST, before building)

1. **Folder placement.** Name is **decided: `scripts/ssrr_ntril/`**. Remaining
   question: keep the SSRR core + `robust_tube_mpc.py` where they are and import
   them, or relocate them under `ssrr_ntril/`?
2. **Damped double integrator.** OK to add a `damping` kwarg to the existing
   `DoubleIntegrator` env (default `0.0`, threaded into `-v0` make-kwargs)?
   Preferred default `c` for experiments? Keep the sinusoidal tracking reference?
3. **Linearized unicycle.** Linearize about (a) a fixed operating point (LTI) or
   (b) a reference trajectory (LTV)? What reference/path (e.g. circle, line) and
   nominal speed? Confirm obs layout (state vs. state+ref) and `Q`/`R`.
4. **Base reward / expert source.** SSRR needs a base reward (DI used AIRL) + a
   suboptimal demonstrator. Per track: reuse the local AIRL path
   (`SSRR/noisy_airl.py`, `tests/test_airl.py`), or pretrained/checkpointed
   SAC/PPO experts (how *suboptimal*: early checkpoint / action noise / capped
   steps)? Any existing MuJoCo checkpoints?
5. **Replication target.** Match SSRR-paper HalfCheetah numbers/protocol
   specifically (which figure/table?), or just "RL on learned reward beats the
   suboptimal expert"? Demos count, noise schedule, snippet config.
6. **Compute / RL budget.** GPU? SAC vs PPO for the RL stage and timestep budget
   (DI used SAC @ 1e5)? Acceptable wall-clock per run?
7. **Dependencies.** Is MuJoCo / `gymnasium[mujoco]` installed and working? (The
   linear track only needs the existing `do_mpc`/`casadi`/`cdd`/`pytope` stack —
   confirm those still import.)
8. **Scope of this first chat.** Just scaffold folders + config + the SSRR runner,
   or push through a first end-to-end SSRR run (and on which track first)?

---

## 9. Suggested first steps (after the user answers)

1. Scaffold the confirmed folder structure + `README.md` (mirror `SSRR/README.md`
   tone) and `common/envs.py` config.
2. **Track A first (fast validation):** add `damping` to the DI env; run the
   existing SSRR pipeline on the *damped* integrator and confirm the reward
   sharpens / SAC tracks better than the undamped case. Then add the linearized
   unicycle env and run SSRR on it. Wire the existing linear tube MPC on both.
3. **Track B:** implement the config-driven `SSRR/runner.py` for HalfCheetah
   (no reference; raw obs+act; `noise_action_scale=1.0`), plus the high-dim
   `reward_eval.py` gate. Keep the affine reward rescale. Run end-to-end; confirm
   RL beats the suboptimal expert.
4. Add `NTRIL/mpc_nonlinear_placeholder.py` boilerplate stub (interface only) so
   the MuJoCo MPC slot exists but is explicitly deferred.

---

### TL;DR for the next agent
Two tracks. **Track A** = damped double integrator + linearized unicycle: linear &
dissipative, **reuse the existing SSRR pipeline, 2-D gate/plots, AND the linear
robust tube MPC**, with damping as the key lever that should make SSRR behave.
**Track B** = MuJoCo (HalfCheetah first): SSRR replication only — **drop reference
tracking**, **add a high-dim return-correlation gate**, keep the affine reward
rescale, and **leave robust MPC as a placeholder stub**. Make the SSRR/RL runner
config-driven across all envs but validate one env per track first. **Ask
Section 8 before creating anything.**

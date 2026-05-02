"""Human-readable run reports for SSRR pipeline stages."""

from __future__ import annotations

from pathlib import Path


def write_rl_run_report(
    *,
    run_dir: Path,
    run_name: str,
    airl_run: str,
    # Environment
    env_id: str,
    max_episode_seconds: float,
    dt: float,
    n_envs: int,
    # Reward regression
    n_ensemble: int,
    reg_lr: float,
    reg_weight_decay: float,
    reg_batch_size: int,
    reg_n_steps: int,
    reg_num_snippets: int,
    # SSRRRegressionConfig
    reg_min_steps: int,
    reg_max_steps: int,
    reg_target_scale: float,
    reg_length_normalize: bool,
    # SAC
    rl_learning_rate: float,
    rl_learning_starts: int,
    rl_buffer_size: int,
    rl_batch_size: int,
    rl_tau: float,
    rl_gamma: float,
    rl_train_freq: int,
    rl_gradient_steps: int,
    rl_total_timesteps: int,
    device: str,
) -> None:
    """Write ``hyperparameters.md`` into *run_dir* capturing the full SSRR RL run config."""
    md = f"""\
# SSRR RL Run — {run_name}

## Lineage
- **AIRL run**: `{airl_run}`
- **Sigmoid fit**: `sigmoid_fit/sigmoid_params.npy`
- **Reward ensemble**: `ssrr_regression/ensemble/` ({n_ensemble} nets)

## Environment
| Parameter | Value |
|-----------|-------|
| `env_id` | `{env_id}` |
| `max_episode_seconds` | `{max_episode_seconds}` |
| `dt` | `{dt}` |
| `n_envs` | `{n_envs}` |

## SSRR Reward Regression
| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_ensemble` | `{n_ensemble}` | Number of independently trained reward nets; ensemble mean used as reward signal |
| `reg_lr` | `{reg_lr}` | Adam learning rate |
| `reg_weight_decay` | `{reg_weight_decay}` | L2 regularisation |
| `reg_batch_size` | `{reg_batch_size}` | Snippet pairs per gradient step |
| `reg_n_steps` | `{reg_n_steps}` | Total gradient steps per ensemble member |
| `reg_num_snippets` | `{reg_num_snippets}` | Total snippet examples drawn from the noisy-rollout buckets |

### SSRRRegressionConfig
| Parameter | Value | Notes |
|-----------|-------|-------|
| `min_steps` | `{reg_min_steps}` | Shortest snippet length sampled (steps) |
| `max_steps` | `{reg_max_steps}` | Longest snippet length sampled (steps) |
| `target_scale` | `{reg_target_scale}` | Scalar multiplied onto the sigmoid target before regression; larger values spread the loss landscape |
| `length_normalize` | `{reg_length_normalize}` | If True, target is `sigma(eta) * (L / T)` — discounts shorter snippets proportionally to their length |

## SAC Policy
| Parameter | Value | Notes |
|-----------|-------|-------|
| `algorithm` | SAC | Soft Actor-Critic (off-policy, continuous actions) |
| `policy` | MlpPolicy | Fully-connected actor/critic |
| `total_timesteps` | `{rl_total_timesteps:,}` | |
| `learning_rate` | `{rl_learning_rate}` | |
| `learning_starts` | `{rl_learning_starts:,}` | Steps of random exploration before gradient updates begin |
| `buffer_size` | `{rl_buffer_size:,}` | Replay buffer capacity |
| `batch_size` | `{rl_batch_size}` | Mini-batch size for each gradient update |
| `tau` | `{rl_tau}` | Soft target-network update coefficient (small = slow-moving target) |
| `gamma` | `{rl_gamma}` | Discount factor |
| `train_freq` | `{rl_train_freq}` | Gradient update every N environment steps |
| `gradient_steps` | `{rl_gradient_steps}` | Gradient updates per `train_freq` cycle |
| `device` | `{device}` | |
"""
    (run_dir / "hyperparameters.md").write_text(md)

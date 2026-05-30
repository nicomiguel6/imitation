# SSRR RL Run — 20260501_221114

## Lineage
- **AIRL run**: `20260420_231943_sinusoidal_A1.0_f0.01`
- **Sigmoid fit**: `sigmoid_fit/sigmoid_params.npy`
- **Reward ensemble**: `ssrr_regression/ensemble/` (3 nets)

## Environment
| Parameter | Value |
|-----------|-------|
| `env_id` | `imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0` |
| `max_episode_seconds` | `1000.0` |
| `dt` | `1.0` |
| `n_envs` | `5` |

## SSRR Reward Regression
| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_ensemble` | `3` | Number of independently trained reward nets; ensemble mean used as reward signal |
| `reg_lr` | `0.0001` | Adam learning rate |
| `reg_weight_decay` | `0.01` | L2 regularisation |
| `reg_batch_size` | `64` | Snippet pairs per gradient step |
| `reg_n_steps` | `3000` | Total gradient steps per ensemble member |
| `reg_num_snippets` | `5000` | Total snippet examples drawn from the noisy-rollout buckets |

### SSRRRegressionConfig
| Parameter | Value | Notes |
|-----------|-------|-------|
| `min_steps` | `50` | Shortest snippet length sampled (steps) |
| `max_steps` | `500` | Longest snippet length sampled (steps) |
| `target_scale` | `10.0` | Scalar multiplied onto the sigmoid target before regression; larger values spread the loss landscape |
| `length_normalize` | `True` | If True, target is `sigma(eta) * (L / T)` — discounts shorter snippets proportionally to their length |

## SAC Policy
| Parameter | Value | Notes |
|-----------|-------|-------|
| `algorithm` | SAC | Soft Actor-Critic (off-policy, continuous actions) |
| `policy` | MlpPolicy | Fully-connected actor/critic |
| `total_timesteps` | `100,000` | |
| `learning_rate` | `0.0003` | |
| `learning_starts` | `10,000` | Steps of random exploration before gradient updates begin |
| `buffer_size` | `1,000,000` | Replay buffer capacity |
| `batch_size` | `256` | Mini-batch size for each gradient update |
| `tau` | `0.01` | Soft target-network update coefficient (small = slow-moving target) |
| `gamma` | `0.99` | Discount factor |
| `train_freq` | `1` | Gradient update every N environment steps |
| `gradient_steps` | `1` | Gradient updates per `train_freq` cycle |
| `device` | `cuda` | |

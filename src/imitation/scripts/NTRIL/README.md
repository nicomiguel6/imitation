# Noisy Trajectory Ranked Imitation Learning (NTRIL)

This directory contains the implementation of the Noisy Trajectory Ranked Imitation Learning (NTRIL) pipeline, which systematically generates and processes demonstration data with varying noise levels to enable robust imitation learning and reward inference.

## Overview

NTRIL is a novel approach to imitation learning that leverages the ranking structure of demonstrations corrupted with different levels of noise to learn more robust reward functions and policies. The pipeline consists of six main steps:

1. **Behavioral Cloning (BC)**: Train an initial policy using standard behavioral cloning
2. **Noisy Rollout Generation**: Generate trajectories with varying noise levels
3. **Robust Tube MPC**: Apply robust control theory to augment the data
4. **Ranked Dataset Construction**: Build a comprehensive ranked dataset
5. **Demonstration Ranked IRL**: Learn reward networks from ranked demonstrations  
6. **Policy Training**: Train the final policy using reinforcement learning

## Key Components

### Core Algorithm (`ntril.py`)
- `NTRILTrainer`: Main class orchestrating the entire NTRIL pipeline
- Handles the complete training workflow from BC to final policy optimization

### Noise Injection (`noise_injection.py`)
- `NoiseInjector`: Configurable noise injection strategies
- `GaussianActionNoiseInjector`: Gaussian noise in action space
- `UniformActionNoiseInjector`: Uniform noise in action space  
- `ParameterNoiseInjector`: Noise in policy parameters
- `NoisyPolicy`: Wrapper for applying noise during rollouts

### Robust Tube MPC (`robust_tube_mpc.py`)
- `RobustTubeMPC`: Implementation of robust tube Model Predictive Control
- Learns linear dynamics from trajectories
- Augments data using Tagliabue's method under disturbance constraints

### Ranked Dataset Construction (`ranked_dataset.py`)
- `RankedDatasetBuilder`: Builds ranked datasets from noisy trajectories
- Supports multiple ranking methods (noise-based, performance-based, hybrid)
- Generates preference pairs for training
- Creates ranking-aware batches for efficient training

### Demonstration Ranked IRL (`demonstration_ranked_irl.py`)
- `DemonstrationRankedIRL`: Learns reward functions from ranked demonstrations
- Uses Bradley-Terry preference model
- Combines ranking loss with preference loss
- Regularization for stable training

### Utilities (`utils.py`)
- Visualization tools for analyzing noise effects
- Ranking quality analysis
- Trajectory diversity metrics
- Noise schedule generation
- Comprehensive reporting

## Usage

### Basic Example

```python
from imitation.scripts.NTRIL import NTRILTrainer
from imitation.data import rollout

# Setup environment and get expert demonstrations
venv = make_vec_env("CartPole-v1", n_envs=1)
expert_trajectories = load_expert_demonstrations()

# Initialize NTRIL trainer
ntril_trainer = NTRILTrainer(
    demonstrations=expert_trajectories,
    venv=venv,
    noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    n_rollouts_per_noise=10,
    mpc_horizon=10,
    disturbance_bound=0.1,
)

# Train the complete pipeline
training_stats = ntril_trainer.train(
    total_timesteps=100000,
    bc_train_kwargs={"n_epochs": 10},
    irl_train_kwargs={"n_epochs": 100},
)

# Get the trained policy
final_policy = ntril_trainer.policy
```

### Advanced Configuration

```python
# Custom noise injection strategy
noise_injector = NoiseInjector(
    strategy="gaussian_action",
    noise_std_scale=0.5,
    clip_actions=True,
)

# Custom MPC parameters
robust_mpc = RobustTubeMPC(
    horizon=20,
    disturbance_bound=0.15,
    tube_radius=0.1,
    linearization_method="finite_difference",
)

# Custom ranking strategy
dataset_builder = RankedDatasetBuilder(
    ranking_method="hybrid",
    preference_noise=0.05,
    min_preference_gap=0.1,
)
```

### Command Line Training

```bash
# Run NTRIL training with Sacred configuration
python -m imitation.scripts.train_ntril with cartpole

# Fast training for testing
python -m imitation.scripts.train_ntril with fast

# MuJoCo environments
python -m imitation.scripts.train_ntril with mujoco environment.gym_id=HalfCheetah-v3
```

## File Structure

```
NTRIL/
├── __init__.py                      # Package initialization
├── guide.md                         # High-level pipeline overview
├── ntril.py                         # Main NTRIL trainer class
├── noise_injection.py               # Noise injection strategies
├── robust_tube_mpc.py              # Robust tube MPC implementation
├── ranked_dataset.py               # Ranked dataset construction
├── demonstration_ranked_irl.py     # Ranked IRL algorithm
└── utils.py                        # Utility functions
```

## Configuration Files

```
scripts/
├── train_ntril.py                  # Training script
└── config/
    └── train_ntril.py              # Sacred configuration
```

## Examples and Tests

```
examples/
└── ntril_example.py                # Simple usage example

tests/
└── test_ntril.py                   # Comprehensive test suite
```

## Key Features

### Robust Data Augmentation
- Uses robust tube MPC to generate additional training data
- Accounts for model uncertainty and disturbances
- Improves sample efficiency

### Ranking-Aware Learning
- Leverages ranking structure in the data
- More stable reward learning compared to standard IRL
- Handles preference ambiguity gracefully

### Flexible Noise Injection
- Multiple noise strategies (action space, parameter space)
- Configurable noise schedules
- Support for curriculum learning

### Comprehensive Evaluation
- Built-in visualization tools
- Ranking quality analysis
- Trajectory diversity metrics
- Detailed reporting

## Dependencies

The NTRIL implementation requires:
- Standard imitation learning dependencies (gymnasium, stable-baselines3, torch)
- Scientific computing libraries (numpy, scipy, sklearn)
- Visualization libraries (matplotlib)
- Sacred for experiment management
- Optional: pytest for testing

## Research Context

NTRIL is designed to address key challenges in imitation learning:
1. **Sample efficiency**: Generate more training data through principled augmentation
2. **Robustness**: Handle noisy or suboptimal demonstrations
3. **Reward learning**: Improve IRL through ranking structure
4. **Scalability**: Work with modern deep RL algorithms

The approach combines ideas from:
- Robust control theory (tube MPC)
- Preference-based learning (Bradley-Terry model) 
- Data augmentation (Tagliabue's method)
- Ranking-based machine learning

## Citation

If you use NTRIL in your research, please cite the relevant papers and this implementation.

"""
Complete NTRIL Example for simple double integrator system

This script demonstrates using the existing NTRILTrainer class from ntril.py
to run the complete NTRIL pipeline on simple double integrator system.

Based on the existing codebase structure in:
- imitation/scripts/NTRIL/ntril.py (NTRILTrainer)
- imitation/scripts/NTRIL/demonstration_ranked_irl.py
- imitation/scripts/NTRIL/noise_injection.py

Author: Nicolas Miguel
Date: December 2025
"""

import os
from pathlib import Path
import functools
import pickle
from typing import Optional, List, Sequence

import numpy as np
import torch as th
import gymnasium as gym
import dataclasses
from stable_baselines3 import PPO
import tqdm

from imitation.algorithms import bc
from imitation.data import rollout, serialize
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import util, logger as imit_logger
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.scripts.NTRIL.noise_injection import EpsilonGreedyNoiseInjector
from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC, RobustTubeMPCPolicy
from imitation.data.types import TrajectoryWithRew
from imitation.data.types import Trajectory


import matplotlib.pyplot as plt


def generate_PID_demonstrations(
    env_id: str = "DoubleIntegrator-v0",
    device: str = "cuda",
    n_episodes: int = 20,
    train_timesteps: int = 10_000_000,
    checkpoint_interval: int = 1_000_000,
    use_checkpoint_at: Optional[int] = None,
    force_retrain: bool = False,
):
    """Generate demonstrations using PID."""
    print("\n" + "=" * 70)
    print("STEP 1a: Generating PID Demonstrations")
    print("=" * 70)

    # Get the directory where THIS script is located
    SCRIPT_DIR = Path(__file__).parent.resolve()

    # Define all paths relative to script directory
    DEBUG_DIR = SCRIPT_DIR / "debug"
    CHECKPOINTS_DIR = DEBUG_DIR / "policy_checkpoints"
    DEMOS_DIR = DEBUG_DIR / "demonstrations"

    DEMO_PATH = DEMOS_DIR / "pid_demonstrations.pkl"

    # Create directories
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)

    ## return if demonstrations already exist
    if DEMO_PATH.exists() and not force_retrain:
        print(f"✓ Found existing PID demonstrations at {DEMOS_DIR / 'pid_demonstrations.pkl'}")
        pid_trajectories = serialize.load(str(DEMO_PATH))
        return pid_trajectories
    else:
        print(f"⚠ No existing PID demonstrations found at {DEMO_PATH}")

    env_pid = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0")

    rng = np.random.default_rng(42)
    pid_trajectories = []
    builder = util.TrajectoryBuilder()
    for j in tqdm.tqdm(range(n_episodes), desc="Generating PID Demonstrations"):
        obs, _ = env_pid.reset()
        builder.start_episode(initial_obs=obs)
        for t in range(200):
            action = env_pid.suboptimal_expert(obs)
            next_obs, reward, _, _, info = env_pid.step(action)
            builder.add_step(action=action, next_obs=next_obs, reward=reward, info=info)
            obs = next_obs
        pid_trajectories.append(builder.finish(terminal=True))
    serialize.save(str(DEMOS_DIR / "pid_demonstrations.pkl"), pid_trajectories)
    print(f"✓ Saved PID demonstrations to {DEMOS_DIR / 'pid_demonstrations.pkl'}")
    env_pid.close()

    return pid_trajectories

def generate_MPC_demonstrations(
    env_id: str = "DoubleIntegrator-v0",
    device: str = "cuda",
    n_episodes: int = 20,
    train_timesteps: int = 10_000_000,
    checkpoint_interval: int = 1_000_000,
    use_checkpoint_at: Optional[int] = None,
    force_retrain: bool = False,
):
    """Generate demonstrations using MPC."""
    print("\n" + "=" * 70)
    print("STEP 1: Generating MPC Demonstrations")
    print("=" * 70)

    # Get the directory where THIS script is located
    SCRIPT_DIR = Path(__file__).parent.resolve()

    # Define all paths relative to script directory
    DEBUG_DIR = SCRIPT_DIR / "debug"
    CHECKPOINTS_DIR = DEBUG_DIR / "policy_checkpoints"
    DEMOS_DIR = DEBUG_DIR / "demonstrations"

    DEMO_PATH = DEMOS_DIR / "mpc_demonstrations.pkl"
    # Create directories
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)

    ## return if demonstrations already exist
    if DEMO_PATH.exists() and not force_retrain:
        print(f"✓ Found existing MPC demonstrations at {DEMOS_DIR / 'mpc_demonstrations.pkl'}")
        mpc_trajectories = serialize.load(str(DEMO_PATH))
        return mpc_trajectories
    else:
        print(f"⚠ No existing MPC demonstrations found at {DEMO_PATH}")

    env_mpc = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0")
    mpc_policy = RobustTubeMPC(
        horizon=10,
        time_step=1.0,
        A=np.array([[0.0, 1.0], [0.0, 0.0]]),
        B=np.array([[0.0], [1.0]]),
        Q=np.diag([10.0, 1.0]),
        R=0.01*np.eye(1),
        state_bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
        control_bounds=(np.array([-50.0]), np.array([50.0])))
    mpc_policy.setup()
    rng = np.random.default_rng()

    # Setup TrajectoryWithRew Types
    # mpc_trajectories = rollout.rollout(
    #     mpc_policy,
    #     venv,
    #     rollout.make_sample_until(min_episodes=10),
    #     rng=rng,
    #     exclude_infos=False,
    # )
    mpc_trajectories = []
    builder = util.TrajectoryBuilder()
    for j in tqdm.tqdm(range(n_episodes), desc="Generating MPC Demonstrations"):
        obs, _ = env_mpc.reset()
        builder.start_episode(initial_obs=obs)
        for t in range(200):
            _, action = mpc_policy.solve_mpc(obs)
            next_obs, reward, _, _, info = env_mpc.step(action)
            builder.add_step(action=action, next_obs=next_obs, reward=reward, info=info)
            obs = next_obs
        mpc_trajectories.append(builder.finish(terminal=True))

    serialize.save(str(DEMOS_DIR / "mpc_demonstrations.pkl"), mpc_trajectories)
    print(f"✓ Saved MPC demonstrations to {DEMOS_DIR / 'mpc_demonstrations.pkl'}")
    env_mpc.close()

    return mpc_trajectories

def train_BC_on_MPC_demonstrations(
    demonstrations: List[TrajectoryWithRew],
    env_id: str = "DoubleIntegrator-v0",
    device: str = "cuda",
    n_episodes: int = 200,
    train_timesteps: int = 10_000_000,
    checkpoint_interval: int = 1_000_000,
    use_checkpoint_at: Optional[int] = None,
    force_retrain: bool = False,
):
    """Train BC policy on MPC demonstrations."""
    print("\n" + "=" * 70)
    print("STEP 2: Training BC Policy on MPC Demonstrations")
    print("=" * 70)

    # Get the directory where THIS script is located
    SCRIPT_DIR = Path(__file__).parent.resolve()

    # Define all paths relative to script directory
    DEBUG_DIR = SCRIPT_DIR / "debug"
    CHECKPOINTS_DIR = DEBUG_DIR / "policy_checkpoints"
    DEMOS_DIR = DEBUG_DIR / "demonstrations"

    # Setup logger
    custom_logger = imit_logger.configure(
        folder=os.path.join(DEBUG_DIR, "logs"),
        format_strs=["stdout", "tensorboard", "csv"],
    )

    rng = np.random.default_rng(42)

    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # Check if BC policy already exists
    BC_POLICY_PATH = CHECKPOINTS_DIR / "bc_mpc_policy.pkl"
    if BC_POLICY_PATH.exists() and not force_retrain:
        # Set up BC policy
        print(f"✓ Found existing BC policy at {BC_POLICY_PATH}")
        bc_policy_temp = bc.reconstruct_policy(str(BC_POLICY_PATH), device=device)
        bc_policy = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            rng=rng,
            demonstrations=demonstrations,
            policy=bc_policy_temp,
            device=device,
            custom_logger=custom_logger,
        )
        print(f"✓ Loaded existing BC policy from {BC_POLICY_PATH}")
    else:
        print(f"⚠ No existing BC policy found at {BC_POLICY_PATH}")
        print("  Training new BC policy...")
        bc_policy = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            rng=rng,
            demonstrations=demonstrations,
            device=device,
            custom_logger=custom_logger,
        )
        bc_policy.train(n_epochs=50, progress_bar=True)
        print(f"✓ Trained new BC policy and saved to {BC_POLICY_PATH}")
        util.save_policy(bc_policy.policy, BC_POLICY_PATH)

    # check if demonstrations already exist
    DEMO_PATH = DEMOS_DIR / "bc_mpc_demonstrations.pkl"
    if DEMO_PATH.exists():
        print(f"✓ Found existing BC MPC demonstrations at {DEMO_PATH}")
        demonstrations = serialize.load(str(DEMO_PATH))
    else:
        print(f"⚠ No existing BC MPC demonstrations found at {DEMO_PATH}")
        demonstrations = rollout.rollout(
            bc_policy.policy,
            venv,
            rollout.make_sample_until(min_episodes=n_episodes),
            rng=rng,
        )
        serialize.save(str(DEMO_PATH), demonstrations)
        print(f"✓ Saved BC MPC demonstrations to {DEMO_PATH}")
        venv.close()

    return bc_policy, demonstrations


def generate_expert_demonstrations(
    env_id: str = "DoubleIntegrator-v0",
    device: str = "cuda",
    n_episodes: int = 20,
    train_timesteps: int = 10_000_000,
    checkpoint_interval: int = 1_000_000,
    use_checkpoint_at: Optional[int] = None,
    force_retrain: bool = False,
):
    """Generate demonstrations using PPO trained to varying degrees of optimality."""
    print("\n" + "=" * 70)
    print("STEP 1: Generating Expert Demonstrations")
    print("=" * 70)

    # Get the directory where THIS script is located
    SCRIPT_DIR = Path(__file__).parent.resolve()

    # Define all paths relative to script directory
    DEBUG_DIR = SCRIPT_DIR / "debug"
    CHECKPOINTS_DIR = DEBUG_DIR / "policy_checkpoints"
    DEMOS_DIR = DEBUG_DIR / "demonstrations"

    # Create directories
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Create environment
    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # Determine which checkpoint to use
    if use_checkpoint_at is None:  # Use final trained policy
        checkpoint_timesteps = train_timesteps
        model_name = f"expert_policy_final_{train_timesteps}"
        MODEL_PATH = CHECKPOINTS_DIR / f"{model_name}.zip"
    else:
        checkpoint_timesteps = use_checkpoint_at
        model_name = f"expert_policy_checkpoint_{use_checkpoint_at}"
        # CheckpointCallback saves as: prefix_<steps>_steps.zip
        MODEL_PATH = (
            CHECKPOINTS_DIR / f"expert_policy_checkpoint_{use_checkpoint_at}_steps.zip"
        )

    DEMO_PATH = DEMOS_DIR / f"{model_name}_demos.pkl"

    print("\nConfiguration:")
    print(f"  Training timesteps: {train_timesteps}")
    print(f"  Checkpoint interval: {checkpoint_interval}")
    print(f"  Using checkpoint at: {checkpoint_timesteps} steps")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Demos path: {DEMO_PATH}")

    # ============================================================
    # CHECK 1: Skip everything if demos already exist
    # ============================================================
    if DEMO_PATH.exists() and not force_retrain:
        print(f"\n✓ Found existing demonstrations at {DEMO_PATH}")
        print(f"  Loading {DEMO_PATH}...")
        expert_trajectories = serialize.load(str(DEMO_PATH))
        print(f"✓ Loaded {len(expert_trajectories)} demonstration trajectories")

        # Display stats
        returns = [sum(traj.rews) for traj in expert_trajectories]
        lengths = [len(traj) for traj in expert_trajectories]
        print(f"  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"  Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")

        venv.close()
        return expert_trajectories

    # ============================================================
    # CHECK 2: Skip training if specific checkpoint exists
    # ============================================================
    if MODEL_PATH.exists() and not force_retrain:
        print(f"\n✓ Found existing checkpoint at {MODEL_PATH}")
        print(f"  Loading policy from checkpoint...")
        expert_policy = PPO.load(str(MODEL_PATH), env=venv, device=device)
        print(f"✓ Loaded policy checkpoint (trained for {checkpoint_timesteps} steps)")
    else:
        # ============================================================
        # TRAIN: No checkpoint exists or force_retrain=True
        # ============================================================
        print(f"\n⚠ Checkpoint not found at {MODEL_PATH}")
        print(
            f"  Training expert policy with checkpoints every {checkpoint_interval} steps..."
        )

        from stable_baselines3.common.callbacks import CheckpointCallback

        # Callback to save model at intervals
        checkpoint_callback = CheckpointCallback(
            save_freq=max(checkpoint_interval // venv.num_envs, 1),
            save_path=str(CHECKPOINTS_DIR),
            name_prefix="expert_policy_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        # Train the policy
        expert_policy = PPO("MlpPolicy", venv, verbose=1, device=device, tensorboard_log=str(CHECKPOINTS_DIR / "ppo_logs"))
        expert_policy.learn(
            total_timesteps=train_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )

        # Save final model
        final_path = CHECKPOINTS_DIR / f"expert_policy_final_{train_timesteps}.zip"
        expert_policy.save(str(final_path))
        print("\n✓ Training complete!")
        print(f"✓ Final policy saved to {final_path}")
        print(f"✓ Checkpoints saved in {CHECKPOINTS_DIR}")

        # List all saved checkpoints
        checkpoints = sorted(
            CHECKPOINTS_DIR.glob("expert_policy_checkpoint_*_steps.zip")
        )
        print(f"\nAvailable checkpoints ({len(checkpoints)}):")
        for ckpt in checkpoints:
            # Extract step number: "expert_policy_checkpoint_10000_steps.zip" -> "10000"
            parts = ckpt.stem.split("_")
            if len(parts) >= 5:
                step_str = parts[-2]  # Second-to-last part is the step number
                print(f"  - {step_str} steps: {ckpt.name}")

        # ============================================================
        # LOAD: After training, load the requested checkpoint
        # ============================================================
        if use_checkpoint_at is not None and use_checkpoint_at < train_timesteps:
            print(f"\n⚠ Requested checkpoint at {use_checkpoint_at} steps")
            if MODEL_PATH.exists():
                print(f"  Loading {MODEL_PATH}...")
                expert_policy = PPO.load(str(MODEL_PATH), env=venv, device=device)
                print("✓ Loaded checkpoint")
            else:
                print("  ERROR: Checkpoint not found after training!")
                print("  Using final trained policy instead.")

    # ============================================================
    # GENERATE DEMONSTRATIONS
    # ============================================================
    print(f"\nGenerating {n_episodes} demonstration trajectories...")
    expert_trajectories = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_episodes=n_episodes),
        rng=rng,
    )

    # Calculate and display trajectory statistics
    returns = [sum(traj.rews) for traj in expert_trajectories]
    lengths = [len(traj) for traj in expert_trajectories]

    print(f"\n✓ Generated {len(expert_trajectories)} demonstration trajectories")
    print(f"  Policy trained for: {checkpoint_timesteps} steps")
    print(f"  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")

    # Save trajectories
    serialize.save(str(DEMO_PATH), expert_trajectories)
    print(f"✓ Saved demonstrations to {DEMO_PATH}")

    venv.close()
    return expert_trajectories


def run_ntril_training(
    demonstrations,
    env_id: str = "DoubleIntegrator-v0",
    save_dir: str = "./ntril_outputs",
    noise_levels: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    n_rollouts_per_noise: int = 10,
    bc_epochs: int = 50,
    rl_total_timesteps: int = 100000,
    run_individual_steps: Optional[list] = None,
    just_plot_noisy_rollouts: bool = False,
    bc_policy: Optional[bc.BC] = None,
    noisy_rollouts: Optional[Sequence[Trajectory]] = None,
    robust_mpc: Optional[RobustTubeMPC] = None,
):
    """Run NTRIL training using the NTRILTrainer class.

    Args:
        demonstrations: Expert demonstration trajectories
        env_id: Gymnasium environment ID
        save_dir: Directory to save outputs
        noise_levels: Sequence of noise levels for data augmentation
        n_rollouts_per_noise: Number of rollouts per noise level
        bc_epochs: Number of epochs for BC training
        rl_total_timesteps: Total timesteps for final RL training
        run_individual_steps: List of steps to run individually
        just_plot_noisy_rollouts: Whether to just plot noisy rollouts (warning, this will override run_individual_steps)
        bc_policy: Optional pre-trained BC policy (needed for steps 2+)
        robust_mpc: Optional robust MPC instance (needed for step 3)

    Returns:
        Trained NTRILTrainer instance
    """
    print("\n" + "=" * 70)
    print("STEP 2: Running NTRIL Training Pipeline")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Create environment
    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # Setup logger
    custom_logger = imit_logger.configure(
        folder=os.path.join(save_dir, "logs"),
        format_strs=["stdout", "tensorboard", "csv"],
    )

    # Create reward network
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
    )

    # Initialize NTRIL trainer
    print("\nInitializing NTRIL Trainer...")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Rollouts per noise level: {n_rollouts_per_noise}")

    ntril_trainer = NTRILTrainer(
        demonstrations=demonstrations,
        venv=venv,
        custom_logger=custom_logger,
        noise_levels=noise_levels,
        n_rollouts_per_noise=n_rollouts_per_noise,
        bc_batch_size=32,
        reward_net=reward_net,
        irl_batch_size=32,
        irl_lr=1e-3,
        save_dir=save_dir,
    )
    
    if bc_policy is not None:
        ntril_trainer.bc_policy = bc_policy.policy
    if robust_mpc is not None:
        ntril_trainer.robust_mpc = robust_mpc

    # Run training
    irl_train_kwargs = {}
    rl_train_kwargs = {}
    
    if just_plot_noisy_rollouts: # hijack the training pipeline to just plot small noisy rollouts
        print("Just plotting noisy rollouts...")
        device = "cuda"
        # Load bc policy
        for noise_level in noise_levels:
            bc_policy = bc.reconstruct_policy(os.path.join(save_dir, "initial_BC_policy"), device=device)
            noisy_policy = EpsilonGreedyNoiseInjector().inject_noise(
                bc_policy, noise_level=noise_level
            )
            rollouts = rollout.rollout(
                noisy_policy, venv, rollout.make_sample_until(min_episodes=2), rng=rng
            )
            plot_noisy_rollouts(noise_level, rollouts)

        return

    if run_individual_steps is None:
        print("\nStarting NTRIL training pipeline...")
        training_stats = ntril_trainer.train(
            total_timesteps=rl_total_timesteps,
            bc_train_kwargs={"n_epochs": bc_epochs, "progress_bar": True},
            irl_train_kwargs=irl_train_kwargs,
            rl_train_kwargs=rl_train_kwargs,
        )

        print("\n" + "=" * 70)
        print("NTRIL Training Complete!")
        print("=" * 70)
        print("\nTraining Statistics:")
        for stage, stats in training_stats.items():
            print(f"\n{stage.upper()}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    else:
        # Run specific steps
        print(f"\nRunning NTRIL steps: {run_individual_steps}")
        step_results = {}

        for step_num in sorted(run_individual_steps):
            print(f"\n{'='*60}")
            print(f"Running Step {step_num}")
            print(f"{'='*60}")

            if step_num == 1: # Trains BC policy from suboptimal expert demonstrations
                print("Step 1: Training BC policy from demonstrations...")
                bc_stats = ntril_trainer._train_bc_policy(
                    n_epochs=bc_epochs, progress_bar=True
                )
                step_results["bc"] = bc_stats
                print("\nBC Training Stats:")
                for key, value in bc_stats.items():
                    print(f"  {key}: {value}")

            elif step_num == 2: # Generates noisy rollouts from BC policy
                # Check if bc policy is provided
                if bc_policy is None:
                    raise ValueError("BC policy is required for step 2")
                print("Step 2: Generating noisy rollouts...")
                _, rollout_stats = ntril_trainer._generate_noisy_rollouts()
                step_results["rollouts"] = rollout_stats
                print(
                    f"\nGenerated rollouts for {len(ntril_trainer.noise_levels)} noise levels"
                )

            elif step_num == 3: # Applies robust tube MPC to augment data
                # Check if robust MPC is provided
                if robust_mpc is None:
                    raise ValueError("Robust MPC is required for step 3")
                # check if noisy rollouts are available at save location
                noisy_rollouts_path = os.path.join(save_dir, "noisy_rollouts.pkl")
                if os.path.exists(noisy_rollouts_path):
                    with open(noisy_rollouts_path, "rb") as f:
                        noisy_rollouts = pickle.load(f)
                else:
                    raise ValueError(f"Noisy rollouts not found at {noisy_rollouts_path}")
                ntril_trainer.noisy_rollouts = noisy_rollouts
                print("Step 3: Augmenting data with robust tube MPC...")

                augmentation_stats = ntril_trainer._augment_data_with_mpc()
                step_results["augmentation"] = augmentation_stats
                print("\nData augmentation complete")

            elif step_num == 4: # Builds ranked dataset
                print("Step 4: Building ranked dataset...")
                ranking_stats = ntril_trainer._build_ranked_dataset()
                step_results["ranking"] = ranking_stats
                print("\nRanked dataset built")

            elif step_num == 5: # Trains reward network using demonstration ranked IRL
                print(
                    "Step 5: Training reward network with demonstration ranked IRL..."
                )
                irl_stats = ntril_trainer._train_reward_network(
                    **(irl_train_kwargs or {})
                )
                step_results["irl"] = irl_stats
                print("\nIRL training complete")

            elif step_num == 6: # Trains final policy using learned reward
                print("Step 6: Training final policy using learned reward...")
                rl_stats = ntril_trainer._train_final_policy(
                    total_timesteps=rl_total_timesteps, **(rl_train_kwargs or {})
                )
                step_results["rl"] = rl_stats
                print("\nFinal policy training complete")

            else:
                raise ValueError(f"Invalid step number: {step_num}. Must be 1-6.")

        print(f"\n{'='*70}")
        print(f"Completed Steps: {run_individual_steps}")
        print(f"{'='*70}")

    venv.close()
    return ntril_trainer


def plot_noisy_rollouts(noise_level, noisy_rollouts, max_rollouts_per_level: int = 2):
    """Plot a few rollouts for each noise level, on separate figures."""
    print("Plotting noisy rollouts...")
    fig, ax = plt.subplots()
    for traj in noisy_rollouts[:max_rollouts_per_level]:
        ax.plot(traj.obs[:, 0], traj.obs[:, 1], label=f"Noise {noise_level}")
        ax.plot(traj.obs[0, 0], traj.obs[0, 1], "b+", label="Initial State")
        ax.plot(traj.obs[-1, 0], traj.obs[-1, 1], "g+", label="Final State")
        ax.grid(True)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
    ax.legend()
    ax.set_title(f"Phase Portrait of Noisy Rollouts at Noise Level {noise_level:.2f}")
    plt.savefig(f"debug/plots/noisy_rollouts_{noise_level}.png")
    plt.close()

    print("Noisy rollouts plotted successfully")

def evaluate_policy(
    policy,
    env_id: str = "DoubleIntegrator-v0",
    n_episodes: int = 10,
):
    """Evaluate a trained policy.

    Args:
        policy: Policy to evaluate
        env_id: Gymnasium environment ID
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 70)
    print("STEP 3: Evaluating Trained Policy")
    print("=" * 70)

    rng = np.random.default_rng(42)

    venv = util.make_vec_env(env_id, rng=rng, n_envs=1)

    eval_trajectories = rollout.rollout(
        policy,
        venv,
        rollout.make_sample_until(min_episodes=n_episodes),
        rng=rng,
    )

    returns = [sum(traj.rews) for traj in eval_trajectories]
    lengths = [len(traj) for traj in eval_trajectories]

    metrics = {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }

    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"  Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")

    venv.close()
    return metrics


def main():
    """Run the complete NTRIL example."""
    # Configuration
    print("Generating expert demonstrations on DoubleIntegrator-v0...")

    # Get the directory where THIS script is located
    SCRIPT_DIR = Path(__file__).parent.resolve()

    SAVE_DIR = SCRIPT_DIR / "ntril_outputs"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    if device == "mps":
        th.set_default_dtype(th.float32)
    else:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

    rngs = np.random.default_rng()
    env_id = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0"

    print("\n" + "=" * 70)
    print("NTRIL PIPELINE FOR DOUBLEINTEGRATOR-V0")
    print("=" * 70)
    print(f"\nEnvironment: {env_id}")

    # Step 1a: Generate MPC demonstrations
    mpc_demonstrations = generate_MPC_demonstrations(
        env_id=env_id,
        device=str(device),
        n_episodes=200,
    )

    # Step 1a: Generate suboptimal demonstrations using PID
    pid_demonstrations = 

    # Step 1b: Train BC policy on MPC demonstrations, rollout policy
    bc_policy, demonstrations = train_BC_on_MPC_demonstrations(
        demonstrations=mpc_demonstrations,
        env_id=env_id,
        device=str(device),
        force_retrain=False,
    )

    '''The use of BC for the MPC demonstrations is not necessary, but it is a good way to get a policy that is close to the MPC policy. 
    Clearly, Actor Critic PPO on the Double Integrator Environment is harder to train to get an expert policy in. 
    Now, we have an "expert policy" that is close to the MPC policy. Now, we need a way to make it suboptimal to represent the true suboptimal policy'''

    # # Step 1: Generate demonstrations (suboptimal)
    # checkpoint_interval = 1_900_000
    # desired_steps = 1_900_000
    # train_timesteps = 2_000_000
    # demonstrations = generate_expert_demonstrations(
    #     env_id=env_id,
    #     device=str(device),
    #     n_episodes=20,
    #     train_timesteps=train_timesteps,  # Train fully
    #     checkpoint_interval=checkpoint_interval,  # Save every 10k steps
    #     use_checkpoint_at=desired_steps,  # Use 30% trained policy (SUBOPTIMAL!)
    #     force_retrain=True,
    # )

    # Step 1.5: Collect reward statistics on suboptimal demonstrations
    print("\nCollecting reward statistics on suboptimal demonstrations...")
    returns = [sum(traj.rews) for traj in demonstrations]
    lengths = [len(traj) for traj in demonstrations]
    print(f"  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")

    # Step 2: Set up robust tube MPC
    robust_tube_mpc = RobustTubeMPC(
        horizon = 10,
        time_step = 1.0,
        disturbance_bound = 0.1,
        tube_radius = 0.05,
        A = np.array([[0.0, 1.0], [0.0, 0.0]]),
        B = np.array([[0.0], [1.0]]),
        Q = np.diag([10.0, 1.0]),
        R = 0.01*np.eye(1),
        disturbance_vertices = np.array([[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1]]),
        state_bounds = (np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
        control_bounds = (np.array([-50.0]), np.array([50.0])),
    )

    robust_tube_mpc.setup()


    # Step 2: Run NTRIL training (Choose step 3 to test sample augmentation)
    ntril_trainer = run_ntril_training(
        demonstrations=demonstrations,
        env_id=env_id,
        save_dir=str(SAVE_DIR),
        noise_levels=(0.0, 0.1),
        n_rollouts_per_noise=10,
        bc_epochs=50,
        rl_total_timesteps=100_000,
        run_individual_steps=[6],  # Change to None to run full training
        just_plot_noisy_rollouts=False,
        bc_policy=bc_policy,
        robust_mpc=robust_tube_mpc,
    )


    # # Step 3: Evaluate the trained policy
    # eval_metrics = evaluate_policy(
    #     policy=ntril_trainer.policy,
    #     env_id=env_id,
    #     n_episodes=10,
    # )
 
    print("\n" + "=" * 70)
    print("NTRIL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {SAVE_DIR}")
    print("\nKey files:")
    print(f"  - Policy checkpoints: {SCRIPT_DIR / 'debug' / 'policy_checkpoints'}")
    print(f"  - Suboptimal demonstrations: {SCRIPT_DIR / 'debug' / 'demonstrations'}")
    print(f"  - Training logs: {SAVE_DIR / 'logs'}")


if __name__ == "__main__":
    main()

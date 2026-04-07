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

import argparse
from datetime import datetime
import os
from pathlib import Path
import functools
import pickle
from typing import Optional, List, Sequence, Dict, Any

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
from imitation.scripts.NTRIL.double_integrator.double_integrator import DoubleIntegratorSuboptimalPolicy
from imitation.scripts.NTRIL.double_integrator.variant_ntril import (
    run_variant_ntril_training,
)
from imitation.policies.base import NonTrainablePolicy
from imitation.scripts.NTRIL.double_integrator.double_integrator import generate_reference_trajectory

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
        n_envs=5,
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
        n_envs=5,
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
    suboptimal_policy: Optional[NonTrainablePolicy] = None,
    demonstrations: Optional[Sequence[TrajectoryWithRew]] = None,
    env_id: str = "DoubleIntegrator-v0",
    env_options: Optional[Dict[str, Any]] = {"max_episode_seconds": 40.0, "dt": 0.1},
    save_dir: str = "./ntril_outputs",
    noise_levels: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    n_rollouts_per_noise: int = 10,
    bc_epochs: int = 50,
    rl_total_timesteps: int = 10000000,
    run_individual_steps: Optional[list] = None,
    retrain: Optional[list] = None,
    just_plot_noisy_rollouts: bool = False,
    noisy_rollouts: Optional[Sequence[Trajectory]] = None,
    robust_mpc: Optional[RobustTubeMPC] = None,
    reference_trajectory: Optional[np.ndarray] = None,
):
    """Run NTRIL training using the NTRILTrainer class.

    Exactly one of ``demonstrations`` or ``bc_policy`` must be provided:

    * ``demonstrations``: raw suboptimal trajectory data; Step 1 will train a
      BC policy on it.
    * ``bc_policy``: a pre-trained suboptimal :class:`bc.BC` instance; Step 1
      is skipped and the policy is used directly for noise injection.

    Args:
        demonstrations: Suboptimal trajectory data (mutually exclusive with
            ``bc_policy``).
        env_id: Gymnasium environment ID.
        env_options: Dictionary of options for the environment.
        save_dir: Directory to save outputs.
        noise_levels: Sequence of noise levels for data augmentation.
        n_rollouts_per_noise: Number of rollouts per noise level.
        bc_epochs: Number of epochs for BC training (ignored when
            ``bc_policy`` is provided).
        rl_total_timesteps: Total timesteps for final RL training.
        run_individual_steps: List of specific pipeline steps to run (1–6).
        retrain: Step names to force-retrain even when cached artefacts exist.
            Accepts ``None`` (use cache), ``"all"``, or a list of names from
            ``("bc", "rollouts", "mpc", "ranking", "irl", "rl")``.
            When ``run_individual_steps`` is used, only the names that map to
            the requested step numbers are applied.
        just_plot_noisy_rollouts: Debug flag to visualise noisy rollouts only.
        noisy_rollouts: Pre-generated noisy rollouts (optional).
        robust_mpc: Robust MPC instance required for Step 3.

    Returns:
        Trained :class:`NTRILTrainer` instance.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Running NTRIL Training Pipeline")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Load reference trajectory
    # reference_trajectory = np.load(os.path.join(save_dir, "reference_trajectory.npy"))
    env_options["reference_trajectory"] = reference_trajectory

    # Create environment
    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
        env_make_kwargs=env_options,
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

    common_kwargs = dict(
        venv=venv,
        custom_logger=custom_logger,
        noise_levels=noise_levels,
        n_rollouts_per_noise=n_rollouts_per_noise,
        bc_batch_size=32,
        reward_net=reward_net,
        irl_batch_size=32,
        irl_lr=1e-3,
        save_dir=save_dir,
        reference_trajectory=reference_trajectory,
    )

    if suboptimal_policy is not None:
        # A pre-trained suboptimal policy is already available — skip BC.
        ntril_trainer = NTRILTrainer.from_policy(suboptimal_policy, **common_kwargs)
    elif demonstrations is not None:
        # Raw trajectory data supplied — BC will be trained in Step 1.
        ntril_trainer = NTRILTrainer.from_demonstrations(demonstrations, **common_kwargs)
    else:
        raise ValueError("Either 'demonstrations' or 'suboptimal_policy' must be provided.")

    if robust_mpc is not None:
        ntril_trainer.robust_mpc = robust_mpc

    # reference_trajectory_mpc = Trajectory(obs=reference_trajectory, acts=np.zeros((venv.envs[0].max_episode_steps, 1)), infos=np.array([{}] * venv.envs[0].max_episode_steps), terminal=True)
    # ntril_trainer.original_reference_trajectory = reference_trajectory_mpc
    
    # Run training
    irl_train_kwargs = {}
    rl_train_kwargs = {}
    
    if just_plot_noisy_rollouts:
        print("Just plotting noisy rollouts...")
        device = "cuda"
        # Resolve the base policy: use the one passed in, or load from disk.
        if suboptimal_policy is not None:
            base_plot_policy = suboptimal_policy
        else:
            base_plot_policy = bc.reconstruct_policy(
                os.path.join(save_dir, "initial_BC_policy"), device=device
            )
        
        # master_rollouts = []
        for noise_level in noise_levels:
            noisy_policy = EpsilonGreedyNoiseInjector().inject_noise(
                base_plot_policy, noise_level=noise_level
            )
            rollouts = rollout.rollout(
                noisy_policy, venv, rollout.make_sample_until(min_episodes=2), rng=rng
            )
            # master_rollouts.append(rollouts[0])

        # # Plot all rollouts together
        # fig, ax = plt.subplots()
        # for rollouts in master_rollouts:
        #     # for traj in rollouts:
        #     ax.plot(rollouts.obs[:, 0])# label=f"Noise {rollouts.infos[0]['noise_level']}")
        # ax.legend()
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Position")
        # ax.set_title("Noisy Rollouts")
        # plt.savefig(os.path.join(save_dir, "noisy_rollouts.png"))
            plot_noisy_rollouts(noise_level, rollouts)

        return

    # Map step numbers to the retrain step names used by NTRILTrainer.
    _STEP_NUM_TO_NAME = {1: "bc", 2: "rollouts", 3: "mpc", 4: "ranking", 5: "irl", 6: "rl"}

    # Resolve force flags for individual steps.
    if retrain == "all":
        _force_set = set(_STEP_NUM_TO_NAME.values())
    elif retrain is None:
        _force_set = set()
    else:
        _force_set = set(retrain)

    if run_individual_steps is None:
        run_individual_steps = [1,2,3,4,5,6]

    # Run specific steps
    print(f"\nRunning NTRIL steps: {run_individual_steps}")
    step_results = {}

    for step_num in sorted(run_individual_steps):
        print(f"\n{'='*60}")
        print(f"Running Step {step_num}")
        print(f"{'='*60}")

        if step_num == 1: # Trains BC policy from suboptimal expert demonstrations
            if suboptimal_policy:
                print("Step 1: Skipping BC policy training as suboptimal policy is provided")
                step_results["bc"] = None
            else:
                print("Step 1: Training BC policy from demonstrations...")
                bc_stats = ntril_trainer._train_bc_policy(
                    force_retrain="bc" in _force_set,
                    n_epochs=bc_epochs, progress_bar=True,
                )
                step_results["bc"] = bc_stats
                print("\nBC Training Stats:")
                for key, value in bc_stats.items():
                    print(f"  {key}: {value}")

        elif step_num == 2: # Generates noisy rollouts from BC/suboptimal policy
            if suboptimal_policy is None:
                raise ValueError("Suboptimal policy is required for step 2")
            print("Step 2: Generating noisy rollouts...")
            noisy_rollouts, rollout_stats = ntril_trainer._generate_noisy_rollouts(
                force_retrain="rollouts" in _force_set,
                # reference_trajectory=reference_trajectory_mpc,
            )
            step_results["rollouts"] = rollout_stats
            print(
                f"\nGenerated rollouts for {len(ntril_trainer.noise_levels)} noise levels"
            )

        elif step_num == 3: # Applies robust tube MPC to augment data
            if robust_mpc is None:
                raise ValueError("Robust MPC is required for step 3")
            noisy_rollouts_path = os.path.join(save_dir, "noisy_rollouts.pkl")
            if os.path.exists(noisy_rollouts_path):
                with open(noisy_rollouts_path, "rb") as f:
                    noisy_rollouts = pickle.load(f)
            else:
                raise ValueError(f"Noisy rollouts not found at {noisy_rollouts_path}")
            ntril_trainer.noisy_rollouts = noisy_rollouts
            print("Step 3: Augmenting data with robust tube MPC...")
            augmentation_data, rtmpc_trajectories = ntril_trainer._augment_data_with_mpc(
                force_retrain="mpc" in _force_set
            )
            step_results["augmentation"] = augmentation_data
            step_results["rtmpc_trajectories"] = rtmpc_trajectories
            print("\nData augmentation complete")

        elif step_num == 4: # Builds ranked dataset
            print("Step 4: Building ranked dataset...")
            ranking_stats = ntril_trainer._build_ranked_dataset(
                force_retrain="ranking" in _force_set
            )
            step_results["ranking"] = ranking_stats
            print("\nRanked dataset built")

        elif step_num == 5: # Trains reward network using demonstration ranked IRL
            print("Step 5: Training reward network with demonstration ranked IRL...")
            irl_stats = ntril_trainer._train_reward_network(
                force_retrain="irl" in _force_set,
                **(irl_train_kwargs or {}),
            )
            step_results["irl"] = irl_stats
            print("\nIRL training complete")

        elif step_num == 6: # Trains final policy using learned reward
            print("Step 6: Training final policy using learned reward...")
            rl_stats = ntril_trainer._train_final_policy(
                total_timesteps=rl_total_timesteps,
                force_retrain="rl" in _force_set,
                **(rl_train_kwargs or {}),
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


def plot_noisy_rollouts(noise_level, noisy_rollouts, max_rollouts_per_level: int = 8):
    """Plot a few rollouts for each noise level, on separate figures."""
    print("Plotting noisy rollouts...")
    fig, ax = plt.subplots()
    # for traj in noisy_rollouts[:max_rollouts_per_level]:
    #     ax.plot(traj.obs[:, 0], traj.obs[:, 1], label=f"Noise {noise_level}")
    #     ax.plot(traj.obs[0, 0], traj.obs[0, 1], "b+", label="Initial State")
    #     ax.plot(traj.obs[-1, 0], traj.obs[-1, 1], "g+", label="Final State")
    #     ax.grid(True)
    #     ax.set_xlabel("Position")
    #     ax.set_ylabel("Velocity")
    #     ax.set_title(f"Phase Portrait of Noisy Rollouts at Noise Level {noise_level:.2f}")
    #     ax.legend()
    # plot as timeseries of position vs time
    for traj in noisy_rollouts[:max_rollouts_per_level]:
        ax.plot(traj.obs[:, 0], label=f"Noise {noise_level}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.set_title(f"Timeseries of Position for Noisy Rollouts at Noise Level {noise_level:.2f}")
        # ax.legend()

    save_dir = "debug/plots/noisy_rollouts"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"noisy_rollouts_{noise_level}.png"))
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


# CLI
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NTRIL training for double integrator.",
    )
    sub = p.add_subparsers(dest="command")
    sp = sub.add_parser("train", help="Train the NTRIL policy")
    # sp.add_argument("--save-dir", type=str, default=str(Path(__file__).parent.resolve() / "ntril_outputs"))
    sp.add_argument("--noise-levels", type=tuple, default=tuple(np.arange(0.0, 1.05, 0.05)))
    sp.add_argument("--n-rollouts-per-noise", type=int, default=5)
    sp.add_argument("--rl-total-timesteps", type=int, default=1_000_000)
    sp.add_argument("--run-individual-steps", type=list, default=[1,2,3,4,5,6])
    sp.add_argument("--retrain", type=list, default=None)
    sp.add_argument("--just-plot-noisy-rollouts", type=bool, default=False)
    sp.add_argument("--archive-name", type=str, default=None,
                    help="Name for the archive directory. Defaults to an auto-generated name: <date>_<mode>_A<amp>_f<freq>.")

    sp = sub.add_parser("plot-noisy-rollouts", help="Plot the noisy rollouts")
    sp.add_argument("--noise-levels", type=tuple, default=tuple(np.arange(0.0, 1.05, 0.05)))
    sp.add_argument("--n-rollouts-per-noise", type=int, default=5)
    sp.add_argument("--rl-total-timesteps", type=int, default=1_000_000)
    sp.add_argument("--run-individual-steps", type=list, default=None)
    sp.add_argument("--retrain", type=list, default=None)
    sp.add_argument("--just-plot-noisy-rollouts", type=bool, default=True)
    sp.add_argument("--archive-name", type=str, default=None)

    sp = sub.add_parser(
        "train-variant",
        help="Train an isolated experimental NTRIL variant",
    )
    sp.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=list(np.arange(0.0, 1.05, 0.05)),
    )
    sp.add_argument("--n-rollouts-per-noise", type=int, default=5)
    sp.add_argument("--n-ensemble", type=int, default=3)
    sp.add_argument("--rl-total-timesteps", type=int, default=1_000_000)
    sp.add_argument("--retrain", nargs="*", default=None)
    sp.add_argument(
        "--ranked-data-source",
        type=str,
        choices=("hybrid", "augmented", "noisy"),
        default="hybrid",
    )
    sp.add_argument("--disable-mpc-step", action="store_true")
    sp.add_argument("--reward-hidden-sizes", type=int, nargs="+", default=[256, 256, 128])
    sp.add_argument(
        "--archive-name",
        type=str,
        default=None,
        help="Name for the variant archive directory.",
    )
    return p


def main():
    """Run the complete NTRIL example."""
    args = _build_parser().parse_args()

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
    max_episode_seconds = 200.0
    dt = 1.0
    ghost_env = gym.make(env_id, max_episode_seconds=max_episode_seconds, dt=dt)

    # Set up reference trajectory and save
    ref_mode = "sinusoidal"
    ref_amplitude = 1.0
    ref_frequency = 0.01
    ref_phase = 0.0
    reference_trajectory = generate_reference_trajectory(
        T=ghost_env.max_episode_steps,
        dt=ghost_env.dt,
        mode=ref_mode,
        amplitude=ref_amplitude,
        frequency=ref_frequency,
        phase=ref_phase,
    )
    reference_trajectory_mpc = Trajectory(obs=reference_trajectory, acts=np.zeros((ghost_env.max_episode_steps, 1)), infos=np.array([{}] * ghost_env.max_episode_steps), terminal=True)
    np.save(SAVE_DIR / "reference_trajectory.npy", reference_trajectory)
    print(f"Saved reference trajectory to {SAVE_DIR / 'reference_trajectory.npy'}")

    _auto_archive_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ref_mode}_A{ref_amplitude}_f{ref_frequency}"

    env_options = {"max_episode_seconds": max_episode_seconds, "dt": dt, "reference_trajectory": reference_trajectory}


    print("\n" + "=" * 70)
    print("NTRIL PIPELINE FOR DOUBLEINTEGRATOR-V0")
    print("=" * 70)
    print(f"\nEnvironment: {env_id}")

    # # Step 1a: Generate MPC demonstrations
    # mpc_demonstrations = generate_MPC_demonstrations(
    #     env_id=env_id,
    #     device=str(device),
    #     n_episodes=200,
    # )

    # # Step 1b: Train BC policy on MPC demonstrations, rollout policy
    # bc_policy, demonstrations = train_BC_on_MPC_demonstrations(
    #     demonstrations=mpc_demonstrations,
    #     env_id=env_id,
    #     device=str(device),
    #     force_retrain=False,
    # )



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

    # # Step 1.5: Collect reward statistics on suboptimal demonstrations
    # print("\nCollecting reward statistics on suboptimal demonstrations...")
    # returns = [sum(traj.rews) for traj in demonstrations]
    # lengths = [len(traj) for traj in demonstrations]
    # print(f"  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    # print(f"  Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")

    # Step 1: Set up robust tube MPC for data augmentation
    robust_tube_mpc = RobustTubeMPC(
        horizon = 10,
        time_step = dt,
        A = ghost_env.A_d,
        B = ghost_env.B_d,
        Q = np.diag([10.0, 1.0]),
        R = 0.1*np.eye(1),
        disturbance_vertices = np.array([[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1]]),
        # artificial_disturbance_vertices = np.array([[-1.5], [1.5]]),
        state_bounds = (np.array([-1000.0, -1000.0]), np.array([1000.0, 1000.0])),
        control_bounds = (np.array([-5.0]), np.array([5.0])),
        reference_trajectory = reference_trajectory_mpc,
        use_approx = True,
    )

    robust_tube_mpc.setup()

    # Step 1.5: Set up suboptimal PID as initial policy
    suboptimal_policy = DoubleIntegratorSuboptimalPolicy(
        observation_space=ghost_env.observation_space,
        action_space=ghost_env.action_space,
    )

    # suboptimal_policy.set_K_values(robust_tube_mpc.K[0,0], robust_tube_mpc.K[0,1])
    K = [0.02, 0.3]
    suboptimal_policy.set_K_values(K[0], K[1])

    if args.command == "train":
        ntril_trainer = run_ntril_training(
            save_dir=str(SAVE_DIR),
            suboptimal_policy=suboptimal_policy,
            env_id=env_id,
            noise_levels=args.noise_levels,
            n_rollouts_per_noise=args.n_rollouts_per_noise,
            rl_total_timesteps=args.rl_total_timesteps,
            run_individual_steps=args.run_individual_steps,
            retrain=args.retrain,
            robust_mpc=robust_tube_mpc,
        )
        archive_name = args.archive_name if args.archive_name else _auto_archive_name
        ntril_trainer.archive_run(
            name=archive_name,
            archive_root=str(SAVE_DIR / "archived_runs"),
        )
    elif args.command == "plot-noisy-rollouts":
        ntril_trainer = run_ntril_training(
            save_dir=str(SAVE_DIR),
            noise_levels=args.noise_levels,
            n_rollouts_per_noise=args.n_rollouts_per_noise,
            rl_total_timesteps=args.rl_total_timesteps,
            run_individual_steps=args.run_individual_steps,
            retrain=args.retrain,
            just_plot_noisy_rollouts=True,
            robust_mpc=robust_tube_mpc,
        )
    # else: # args.command == "train-variant":
    #     variant_save_dir = SCRIPT_DIR / "variant_outputs"
    #     # variant_retrain = args.retrain
    #     variant_retrain = ["ranking", "irl", "rl"]
    #     if variant_retrain == ["all"]:
    #         variant_retrain = "all"
    #     elif variant_retrain == []:
    #         variant_retrain = None
    #     # variant_kwargs = {
    #     #     "ranked_data_source": args.ranked_data_source,
    #     #     "include_mpc_step": not args.disable_mpc_step,
    #     #     "reward_hidden_sizes": tuple(args.reward_hidden_sizes),
    #     # }
    #     # ntril_trainer, variant_ref_tag = run_variant_ntril_training(
    #     #     env_id=env_id,
    #     #     env_options={"max_episode_seconds": max_episode_seconds, "dt": dt},
    #     #     save_dir=str(variant_save_dir),
    #     #     noise_levels=tuple(args.noise_levels),
    #     #     n_rollouts_per_noise=args.n_rollouts_per_noise,
    #     #     n_ensemble=args.n_ensemble,
    #     #     rl_total_timesteps=args.rl_total_timesteps,
    #     #     retrain=variant_retrain,
    #     #     variant_kwargs=variant_kwargs,
    #     # )
    #     # archive_name = (
    #     #     args.archive_name
    #     #     if args.archive_name
    #     #     else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{variant_ref_tag}_variant"
    #     # )
    #     variant_kwargs = {
    #         "ranked_data_source": "hybrid",
    #         "include_mpc_step": True,
    #         "reward_hidden_sizes": (256, 256),
    #     }
    #     ntril_trainer = run_variant_ntril_training(
    #         suboptimal_policy=suboptimal_policy,
    #         env_id=env_id,
    #         env_options=env_options,
    #         save_dir=str(variant_save_dir),
    #         noise_levels=tuple(np.arange(0.0, 1.05, 0.05)),
    #         n_rollouts_per_noise=5,
    #         n_ensemble=3,
    #         rl_total_timesteps=1_000_000,
    #         run_individual_steps=[1,2,3,4,5,6],
    #         retrain=variant_retrain,
    #         variant_kwargs=variant_kwargs,
    #         robust_mpc=robust_tube_mpc,
    #         reference_trajectory=reference_trajectory,
    #     )
    #     archive_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_variant"
    #     ntril_trainer.archive_run(
    #         name=archive_name,
    #         archive_root=str(variant_save_dir / "archived_runs"),
    #     )
    else:
        ntril_trainer = run_ntril_training(
            suboptimal_policy=suboptimal_policy,
            env_id=env_id,
            env_options=env_options,
            save_dir=str(SAVE_DIR),
            noise_levels=tuple(np.arange(0.0, 1.05, 0.05)),
            n_rollouts_per_noise=8,
            rl_total_timesteps=1_000_000,
            run_individual_steps=[2,3,4,5,6],
            retrain=["rollouts", "mpc", "ranking", "irl", "rl"],
            # retrain=None,
            just_plot_noisy_rollouts=False,
            robust_mpc=robust_tube_mpc,
            reference_trajectory=reference_trajectory,
        )
        ntril_trainer.archive_run(
            name=_auto_archive_name,
            archive_root=str(SAVE_DIR / "archived_runs"),
        )




    # # Plot an instance of a nominal noisy rollout vs rtmpc trajectory for each noise level
    # mpc_plot_dir = SCRIPT_DIR / "debug" / "plots" / "mpc"
    # mpc_plot_dir.mkdir(parents=True, exist_ok=True)
    # for idx, noise_level in enumerate(ntril_trainer.noise_levels):
    #     nominal_noisy_rollout = ntril_trainer.noisy_rollouts[idx][0]
    #     rtmpc_trajectory = ntril_trainer.rtmpc_trajectories[idx][0]
        
    #     fig, ax = plt.subplots()
    #     ax.plot(nominal_noisy_rollout.obs[:, 0], label="Nominal Noisy Rollout")
    #     ax.plot(rtmpc_trajectory.obs[:, 0], label="RTMPC Trajectory")
    #     # sdet adjusted state bounds
    #     ax.axhline(y=-robust_tube_mpc.b_x_t[1], color="r", linestyle="--", linewidth=1)
    #     ax.axhline(y=robust_tube_mpc.b_x_t[3], color="r", linestyle="--", linewidth=1)
    #     ax.legend()
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel("Position")
    #     ax.set_title(f"Nominal Noisy Rollout vs RTMPC Trajectory at Noise Level {noise_level:.2f}")
    #     plt.savefig(mpc_plot_dir / f"nominal_comparison_{noise_level:.2f}.png")
    #     plt.close()
    #     print(f"Plotted nominal noisy rollout vs rtmpc trajectory for noise level {noise_level:.2f}")

    # Plot RTMPC trajectory and augmented data for a noise level
    augmented_plots_dir = SCRIPT_DIR / "debug" / "plots" / "augmented_plots"
    augmented_plots_dir.mkdir(parents=True, exist_ok=True)

    for idx, noise_level in enumerate(ntril_trainer.noise_levels):
        # Use the first RTMPC trajectory and all augmented data for each noise level
        rtmpc_trajectory = ntril_trainer.rtmpc_trajectories[idx][0]  # one trajectory at this noise level
        # if noise_level == 0.0:
        #     augmented_data = ntril_trainer.augmented_data[idx]           # augmented trajectories at this noise level
        # else:
            # choose 10 random augmented trajectories at this noise level   
        chosen_indices = np.random.choice(len(ntril_trainer.augmented_data[idx]), 10, replace=False)
        augmented_data = [ntril_trainer.augmented_data[idx][int(i)] for i in chosen_indices]

        fig, ax = plt.subplots()
        ax.plot(rtmpc_trajectory.obs[:, 0], color="blue", label="RTMPC Trajectory")
        for i, traj in enumerate(augmented_data):
            label = "Augmented Data" if i == 0 else None
            timesteps = [traj.infos[i]["current_timestep"] for i in range(len(traj.obs)-1)]
            ax.plot(timesteps, traj.obs[:-1, 0], color="red", label=label)
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.set_title(f"RTMPC Trajectory and Augmented Data at Noise Level {noise_level:.2f}")
        plt.savefig(augmented_plots_dir / f"rtmpc_trajectory_{noise_level:.2f}.png")
        plt.close(fig)


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

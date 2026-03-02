"""Main NTRIL (Noisy Trajectory Ranked Imitation Learning) algorithm implementation."""

import dataclasses
import json
import pickle
import logging
import os
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Mapping, Optional, Sequence, Union, List

import numpy as np
import torch as th
from torch.utils.data import DataLoader
from stable_baselines3.common import policies, vec_env
from stable_baselines3 import PPO

from imitation.algorithms import base, bc
from imitation.data import rollout, types, serialize
from imitation.rewards.reward_nets import RewardNet, TrajectoryRewardNet
from imitation.scripts.NTRIL.noise_injection import EpsilonGreedyNoiseInjector
from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.scripts.NTRIL.demonstration_ranked_irl import RankedTransitionsDataset, DemonstrationRankedIRL
from imitation.util import logger as imit_logger
from imitation.util import util

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

# Type alias for the two supported suboptimal expert inputs.
SuboptimalExpert = Union[Sequence[types.Trajectory], policies.BasePolicy]


class _ExpertMode(Enum):
    DEMONSTRATIONS = auto()  # trajectories provided; BC will be trained
    POLICY = auto()          # policy provided directly; BC step is skipped


@dataclasses.dataclass
class NTRILTrainer(base.BaseImitationAlgorithm):
    """Noisy Trajectory Ranked Imitation Learning trainer.

    Do not instantiate directly.  Use one of the two factory class-methods:

    * ``NTRILTrainer.from_demonstrations(demonstrations, venv, ...)``
      – Step 1 trains a BC policy on the supplied trajectory data.

    * ``NTRILTrainer.from_policy(policy, venv, ...)``
      – Step 1 is skipped; the supplied policy is used as the initial
        suboptimal policy for noise injection.
    """

    # --- internal fields set by the factory constructors ---
    # Exactly one of (_demonstrations, _suboptimal_policy) is not None.
    _demonstrations: Optional[Sequence[types.Trajectory]]
    _suboptimal_policy: Optional[policies.BasePolicy]
    _expert_mode: _ExpertMode

    venv: vec_env.VecEnv
    custom_logger: Optional[imit_logger.HierarchicalLogger] = None

    # Training configuration
    noise_levels: Sequence[float] = dataclasses.field(
        default_factory=lambda: (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    )
    n_rollouts_per_noise: int = 10
    mpc_horizon: int = 10
    disturbance_bound: float = 0.1
    bc_batch_size: int = 32
    bc_train_kwargs: Optional[Mapping[str, Any]] = None
    reward_net: Optional[RewardNet] = None
    irl_batch_size: int = 32
    irl_lr: float = 1e-3
    save_dir: Optional[str] = None
    rng: int = 42
    n_ensemble: int = 3

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_demonstrations(
        cls,
        demonstrations: Sequence[types.Trajectory],
        venv: vec_env.VecEnv,
        **kwargs,
    ) -> "NTRILTrainer":
        """Create a trainer that learns a BC policy from trajectory data.

        Step 1 of the NTRIL pipeline will train a BC policy on
        ``demonstrations``.  All subsequent steps proceed normally.

        Args:
            demonstrations: Suboptimal trajectory data to clone.
            venv: Vectorised training environment.
            **kwargs: Forwarded to :class:`NTRILTrainer`.
        """
        return cls(
            _demonstrations=demonstrations,
            _suboptimal_policy=None,
            _expert_mode=_ExpertMode.DEMONSTRATIONS,
            venv=venv,
            **kwargs,
        )

    @classmethod
    def from_policy(
        cls,
        policy: policies.BasePolicy,
        venv: vec_env.VecEnv,
        **kwargs,
    ) -> "NTRILTrainer":
        """Create a trainer that uses a pre-existing suboptimal policy.

        Step 1 of the NTRIL pipeline is skipped; ``policy`` is used directly
        as the base policy for noise injection.

        Args:
            policy: A ``stable_baselines3`` compatible policy that represents
                the suboptimal expert.
            venv: Vectorised training environment.
            **kwargs: Forwarded to :class:`NTRILTrainer`.
        """
        return cls(
            _demonstrations=None,
            _suboptimal_policy=policy,
            _expert_mode=_ExpertMode.POLICY,
            venv=venv,
            **kwargs,
        )

    def __post_init__(self):
        """Initialize components after dataclass creation."""
        # The dataclass-generated __init__ does not call super().__init__(), so
        # BaseImitationAlgorithm.__init__ (which sets _logger) is never reached
        # automatically.  We call it explicitly here so that _logger is always
        # available throughout the class hierarchy (including DREXTrainer).
        super().__init__(custom_logger=self.custom_logger)

        # Device setup
        if th.cuda.is_available():
            device = th.device("cuda:0")
        else:
            device = th.device("cpu")
        object.__setattr__(self, 'device', device)

        # Save dir setup
        if self.save_dir is None:
            save_dir = os.path.join(
                os.getcwd(), "ntril_runs", datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        else:
            save_dir = os.path.abspath(self.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        object.__setattr__(self, 'save_dir', save_dir)

        object.__setattr__(self, 'rng', np.random.default_rng(self.rng))

        # BC trainer — only created when we have demonstrations to train on.
        if self._expert_mode is _ExpertMode.DEMONSTRATIONS:
            bc_trainer = bc.BC(
                observation_space=self.venv.observation_space,
                action_space=self.venv.action_space,
                policy=None,
                demonstrations=self._demonstrations,
                batch_size=self.bc_batch_size,
                device=device,
                custom_logger=self.custom_logger,
                rng=self.rng,
                **(self.bc_train_kwargs or {}),
            )
            object.__setattr__(self, 'bc_trainer', bc_trainer)
            object.__setattr__(self, 'bc_policy', None)
        else:
            # Policy supplied directly: pre-populate bc_policy and skip BC.
            object.__setattr__(self, 'bc_trainer', None)
            object.__setattr__(self, 'bc_policy', self._suboptimal_policy)

        # Reward network
        if self.reward_net is None:
            reward_net = TrajectoryRewardNet(
                observation_space=self.venv.observation_space,
                action_space=self.venv.action_space,
                use_state=True,
                use_action=False,
                use_next_state=False,
                use_done=False,
            )
            object.__setattr__(self, 'reward_net', reward_net)

        # Move reward net to device
        self.reward_net.to(device)

        # Storage for training artifacts
        object.__setattr__(self, 'noisy_rollouts', [])
        object.__setattr__(self, 'augmented_data', [])
        object.__setattr__(self, 'ranked_dataset', None)
        object.__setattr__(self, 'ranked_datasets', [])
        object.__setattr__(self, 'reward_nets_ensemble', [])
        object.__setattr__(self, 'learned_reward_net', None)
        object.__setattr__(self, 'final_policy', None)

    #: Names of all pipeline steps, in execution order.
    STEPS: tuple = ("bc", "rollouts", "mpc", "ranking", "irl", "rl")

    def train(
        self,
        total_timesteps: int,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
        irl_train_kwargs: Optional[Mapping[str, Any]] = None,
        rl_train_kwargs: Optional[Mapping[str, Any]] = None,
        retrain: Union[str, Sequence[str], None] = None,
    ) -> Dict[str, Any]:
        """Run the complete NTRIL training pipeline.

        Args:
            total_timesteps: Total timesteps for final RL training.
            bc_train_kwargs: Additional kwargs for BC training.
            irl_train_kwargs: Additional kwargs for IRL training.
            rl_train_kwargs: Additional kwargs for RL training.
            retrain: Which pipeline steps to force-retrain even when cached
                artefacts already exist on disk.  Accepts:

                * ``None`` (default) – use cached results for every step.
                * ``"all"`` – force every step to retrain.
                * A sequence of step names chosen from
                  ``("bc", "rollouts", "mpc", "ranking", "irl", "rl")``.

                Example – retrain only the reward network and final policy::

                    trainer.train(1_000_000, retrain=["irl", "rl"])

        Returns:
            Dictionary containing training statistics.
        """
        if retrain == "all":
            force = set(self.STEPS)
        elif retrain is None:
            force: set = set()
        else:
            unknown = set(retrain) - set(self.STEPS)
            if unknown:
                raise ValueError(
                    f"Unknown step name(s) in retrain: {unknown}. "
                    f"Valid names are: {self.STEPS}"
                )
            force = set(retrain)

        stats = {}

        # Step 1: Train BC policy
        self._logger.log("Starting behavioral cloning training...")
        bc_stats = self._train_bc_policy(force_retrain="bc" in force, **(bc_train_kwargs or {}))
        stats["bc"] = bc_stats

        # Step 2: Generate noisy rollouts
        self._logger.log("Generating noisy rollouts...")
        rollout_stats = self._generate_noisy_rollouts(force_retrain="rollouts" in force)
        stats["rollouts"] = rollout_stats

        # Step 3: Apply robust tube MPC and augment data
        self._logger.log("Applying robust tube MPC and augmenting data...")
        augmentation_stats = self._augment_data_with_mpc(force_retrain="mpc" in force)
        stats["augmentation"] = augmentation_stats

        # Step 4: Build ranked dataset
        self._logger.log("Building ranked dataset...")
        ranking_stats = self._build_ranked_dataset(force_retrain="ranking" in force)
        stats["ranking"] = ranking_stats

        # Step 5: Train reward network using demonstration ranked IRL
        self._logger.log("Training reward network with demonstration ranked IRL...")
        irl_stats = self._train_reward_network(force_retrain="irl" in force, **(irl_train_kwargs or {}))
        stats["irl"] = irl_stats

        # Step 6: Train final policy using RL
        self._logger.log("Training final policy using RL...")
        rl_stats = self._train_final_policy(total_timesteps, force_retrain="rl" in force, **(rl_train_kwargs or {}))
        stats["rl"] = rl_stats

        return stats

    def _train_bc_policy(self, force_retrain: bool = False, **kwargs) -> Dict[str, Any]:
        """Obtain the initial suboptimal policy (Step 1).

        * **Demonstrations mode**: trains (or loads a cached) BC policy from
          the supplied trajectory data.
        * **Policy mode**: the policy was supplied at construction time, so
          this step only evaluates it and returns statistics.

        Args:
            force_retrain: If ``True``, retrain the BC policy even if a cached
                policy already exists on disk.
        """
        save_initial_bc_policy_dir = os.path.join(self.save_dir, "initial_BC_policy")
        object.__setattr__(self, 'save_initial_bc_policy_dir', save_initial_bc_policy_dir)

        if self._expert_mode is _ExpertMode.POLICY:
            # Policy was provided directly — nothing to train.
            logger.info("Skipping BC training: suboptimal policy was provided directly.")
        else:
            # Demonstrations mode: train or load a cached BC policy.
            if os.path.exists(save_initial_bc_policy_dir) and not force_retrain:
                print("Loading existing BC policy...")
                self.bc_policy = bc.reconstruct_policy(
                    save_initial_bc_policy_dir, device=self.device)
            else:
                print("Training BC policy...")
                self.bc_trainer.train(**kwargs)
                self.bc_policy = self.bc_trainer.policy

            util.save_policy(self.bc_policy, save_initial_bc_policy_dir)

        # Evaluate the policy regardless of how it was obtained.
        bc_rollouts = rollout.rollout(
            self.bc_policy,
            self.venv,
            rollout.make_sample_until(min_timesteps=1000),
            rng=self.rng,
        )

        return {
            "n_rollouts": len(bc_rollouts),
            "mean_return": np.mean([sum(traj.rews) for traj in bc_rollouts]),
            "mean_length": np.mean([len(traj) for traj in bc_rollouts]),
        }

    def _generate_noisy_rollouts(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Generate rollouts with different noise levels.

        Args:
            force_retrain: If ``True``, regenerate rollouts even if a cached
                file already exists on disk.
        """
        if self.bc_policy is None:
            raise ValueError(
                "BC policy must be trained/loaded before generating noisy rollouts"
            )

        noisy_rollouts_path = os.path.join(self.save_dir, "noisy_rollouts.pkl")
        if os.path.exists(noisy_rollouts_path) and not force_retrain:
            print("Loading existing noisy rollouts...")
            with open(noisy_rollouts_path, "rb") as f:
                self.noisy_rollouts = pickle.load(f)
            total_rollouts = sum(len(r) for r in self.noisy_rollouts)
            return self.noisy_rollouts, {
                "total_rollouts": total_rollouts,
                "noise_levels": list(self.noise_levels),
                "rollouts_per_level": [len(r) for r in self.noisy_rollouts],
            }

        self.noisy_rollouts = []
        total_rollouts = 0
        self.noise_injector = EpsilonGreedyNoiseInjector()
        noisy_policies_dir = os.path.join(self.save_dir, "noisy_policies")
        os.makedirs(noisy_policies_dir, exist_ok=True)

        base_policy_path = os.path.join(noisy_policies_dir, "base_policy.pt")
        if not os.path.exists(base_policy_path) or force_retrain:
            util.save_policy(self.bc_policy, base_policy_path)

        noisy_policies_metadata = []

        noise_rollout_data = {}

        for noise_level in self.noise_levels:
            # Create noisy policy
            noisy_policy = self.noise_injector.inject_noise(
                self.bc_policy, noise_level=noise_level
            )
            noisy_policies_metadata.append(
                {
                    "noise_level": float(noise_level),
                    "base_policy_path": base_policy_path,
                    "noise_injector": type(self.noise_injector).__name__,
                }
            )

            # Collect rollouts (exclude_infos=False so label_info / noise_applied stay)
            rollouts = rollout.rollout(
                noisy_policy,
                self.venv,
                rollout.make_sample_until(
                    min_episodes=self.n_rollouts_per_noise,
                    min_timesteps=1000,
                ),
                rng=self.rng,
                label_info={"noise_level": noise_level},
                exclude_infos=False,
            )

            self.noisy_rollouts.append(rollouts)
            total_rollouts += len(rollouts)

            # self._logger.log(
            #     f"Collected {len(rollouts)} rollouts with noise level {noise_level}"
            # )

            mean_reward = np.mean([sum(traj.rews) for traj in rollouts])
            std_reward = np.std([sum(traj.rews) for traj in rollouts])
            noise_rollout_data[noise_level] = (mean_reward, std_reward)

        metadata_path = os.path.join(noisy_policies_dir, "noisy_policies.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(noisy_policies_metadata, f, indent=2)
        
        # Save noisy rollouts to file
        noisy_rollouts_path = os.path.join(self.save_dir, "noisy_rollouts.pkl")
        with open(noisy_rollouts_path, "wb") as f:
            pickle.dump(self.noisy_rollouts, f)

        plot_path = os.path.join(self.save_dir, "noise_levels_visualization.png")

        # Visualize noise levels in rollouts with std deviation plot
        plt.errorbar(self.noise_levels, 
                     [noise_rollout_data[n][0] for n in self.noise_levels],
                     yerr=[noise_rollout_data[n][1] for n in self.noise_levels],
                     fmt='-o')
        plt.xlabel('Noise Level (Epsilon)')
        plt.ylabel('Mean Return')
        plt.title('Noisy Rollouts Performance')
        plt.savefig(plot_path)
        plt.close()

        return self.noisy_rollouts, {
            "total_rollouts": total_rollouts,
            "noise_levels": list(self.noise_levels),
            "rollouts_per_level": [len(rollouts) for rollouts in self.noisy_rollouts],
        }

    def _augment_data_with_mpc(self, force_retrain: bool = False) -> List[List[List[types.TrajectoryWithRew]]]:
        """Apply robust tube MPC to augment the noisy rollouts.
        Suppose we have S different noise levels. 
        For each noise level, we have K base trajectory rollouts of length T resulting from RTMPC.
        For each trajectory rollout, we select T/k_timesteps steps to sample from.
        At each step, we get N_samples from the tube surrounding the nominal state of the trajectory.
        For each sample, we propagate the dynamics starting from that sample and following the ancillary controller u = u0 + K(x-x0) .
        Thefore, for each trajectory rollout, we get K*N_samples*T/k_timesteps augmented trajectories.

        EXAMPLE:
        Suppose we have 5 noise levels, and for each noise level, we have 2 trajectory rollouts of length 100.
        We sample every 10 steps for each trajectory rollout, and produce 4 samples per step.
        For each sample, we propagate forward 20 steps using the dynamics and the ancillary controller.
        Therefore, for each noise level, we have 2*(100/10)*4 = 80 augmented trajectory snippets.

        STRUCTURE OF AUGMENTED DATA:
        Augmented data is a list of lists, where each inner list contains the augmented trajectories for a given noise level.
        self.augmented_data = [[augmented_trajectories_noise_level_1],
                               [augmented_trajectories_noise_level_2],
                               ...,
                               [augmented_trajectories_noise_level_S]]
        
        augmented_trajectories_noise_level_i = [augmented_trajectory_base_1, augmented_trajectory_base_2, ..., augmented_trajectory_base_K] (list of K original rollouts that were augmented)
        augmented_trajectory_base_j = [augmented_trajectory_base_j_1, augmented_trajectory_base_j_2, ..., augmented_trajectory_base_j_N_samples*T/k_timesteps] (list of N_samples*T/k_timesteps augmented trajectories for the j-th original rollout)

        So self.augmented_data[i][j] is the j-th augmented trajectory for the i-th noise level.
        """
        self.rtmpc_trajectories = []
        self.augmented_data = []

        # Check if augmented data and rtmpc trajectories already exists in the save directory
        augmented_data_path = os.path.join(self.save_dir, "augmented_data")
        rtmpc_trajectories_path = os.path.join(self.save_dir, "rtmpc_trajectories")
        if os.path.exists(augmented_data_path) and os.path.exists(rtmpc_trajectories_path) and not force_retrain:
            for noise_level in self.noise_levels:
                augmented_data_for_noise_level_path = os.path.join(augmented_data_path, f"noise_{noise_level:.2f}.pkl")
                rtmpc_trajectories_for_noise_level_path = os.path.join(rtmpc_trajectories_path, f"noise_{noise_level:.2f}.pkl")
                if os.path.exists(augmented_data_for_noise_level_path) and os.path.exists(rtmpc_trajectories_for_noise_level_path):
                    augmented_data_for_noise_level = serialize.load(augmented_data_for_noise_level_path)
                    rtmpc_trajectories_for_noise_level = serialize.load(rtmpc_trajectories_for_noise_level_path)
                    self.augmented_data.append(augmented_data_for_noise_level)
                    self.rtmpc_trajectories.append(rtmpc_trajectories_for_noise_level)
            return self.augmented_data, self.rtmpc_trajectories


        for noise_idx, rollouts in enumerate(self.noisy_rollouts):
            noise_level = self.noise_levels[noise_idx]
            augmented_data_for_noise_level = []
            rtmpc_trajectories_for_noise_level = []

            for traj_idx, traj in enumerate(rollouts):
                # Apply robust tube MPC to each trajectory to get nominal trajectory
                self.robust_mpc.set_reference_trajectory(traj)
                rtmpc_trajectory = self._solve_rtmpc(traj.obs[0], traj)
                rtmpc_trajectories_for_noise_level.append(rtmpc_trajectory)

                # Augment from nominal trajectory to get augmented trajectories
                augmented_trajectories = self.robust_mpc.augment_trajectory(
                    rtmpc_trajectory
                )
                augmented_data_for_noise_level.extend(augmented_trajectories)
            
            # Save rtmpc_trajectories_for_noise_level to a file
            serialize.save(os.path.join(self.save_dir, "rtmpc_trajectories", f"noise_{noise_level:.2f}.pkl"), rtmpc_trajectories_for_noise_level)
            self.rtmpc_trajectories.append(rtmpc_trajectories_for_noise_level)
            # Save each augmented_data_for_noise_level to a file
            serialize.save(os.path.join(self.save_dir, "augmented_data", f"noise_{noise_level:.2f}.pkl"), augmented_data_for_noise_level)
            self.augmented_data.append(augmented_data_for_noise_level) 
        
        return self.augmented_data, self.rtmpc_trajectories

    def _build_ranked_dataset(self, force_retrain: bool = False) -> List[RankedTransitionsDataset]:
        """Build K ranked datasets from the input trajectories (Step 4).

        Each of the ``n_ensemble`` datasets is drawn with a different random
        seed, giving ensemble members independent training data.  Files are
        cached under ``<save_dir>/ensemble/ranked_samples_{i}.pth``.

        The data source is determined by the subclass:
        * :class:`NTRILTrainer` uses MPC-augmented trajectories
          (``self.augmented_data``).
        * :class:`DREXTrainer` uses the noisy rollouts directly
          (``self.noisy_rollouts``).

        Args:
            force_retrain: Rebuild all datasets even when cached files exist.

        Returns:
            List of :class:`RankedTransitionsDataset` (length ``n_ensemble``).
        """
        ensemble_dir = os.path.join(self.save_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)

        cache_paths = [
            os.path.join(ensemble_dir, f"ranked_samples_{i}.pth")
            for i in range(self.n_ensemble)
        ]
        all_cached = all(os.path.exists(p) for p in cache_paths)

        if all_cached and not force_retrain:
            self.ranked_datasets = []
            for path in cache_paths:
                saved = th.load(path)
                self.ranked_datasets.append(
                    RankedTransitionsDataset(
                        demonstrations=None,
                        training_samples=saved["samples"],
                        num_snippets=saved["num_snippets"],
                        min_segment_length=saved["min_segment_length"],
                        max_segment_length=saved["max_segment_length"],
                    )
                )
            self.ranked_dataset = self.ranked_datasets[0]
            return self.ranked_datasets

        # Resolve the data source: augmented_data for NTRIL, noisy_rollouts for DREX.
        data_source = self._get_ranked_dataset_source()

        self.ranked_datasets = []
        for i in range(self.n_ensemble):
            dataset = RankedTransitionsDataset(
                demonstrations=data_source,
                num_snippets=5_000,
                min_segment_length=5,
                max_segment_length=30,
                rng=np.random.default_rng(i),
            )
            th.save(
                {
                    "samples": [dataset[j] for j in range(len(dataset))],
                    "num_snippets": dataset.num_snippets,
                    "min_segment_length": dataset.min_segment_length,
                    "max_segment_length": dataset.max_segment_length,
                },
                cache_paths[i],
            )
            self.ranked_datasets.append(dataset)

        self.ranked_dataset = self.ranked_datasets[0]
        return self.ranked_datasets

    def _get_ranked_dataset_source(self) -> list:
        """Return the trajectory data used to build the ranked dataset.

        :class:`NTRILTrainer` uses MPC-augmented trajectories.
        Subclasses can override this to supply a different data source
        (e.g. :class:`DREXTrainer` returns ``self.noisy_rollouts``).
        """
        if not self.augmented_data:
            for noise_level in self.noise_levels:
                path = os.path.join(
                    self.save_dir, "augmented_data", f"noise_{noise_level:.2f}.pkl"
                )
                if os.path.exists(path):
                    self.augmented_data.append(serialize.load(path))
                else:
                    raise ValueError(
                        f"Augmented data not found at {path}. "
                        "Run _augment_data_with_mpc() first."
                    )
        return self.augmented_data

    def _train_reward_network(self, force_retrain: bool = False, **kwargs) -> List[TrajectoryRewardNet]:
        """Train one reward network per ranked dataset (Step 5).

        Defaults match the D-REX paper (Brown et al., 2020):
        1 000 optimizer steps, lr=1e-4, batch_size=64, weight_decay=0.01.
        Networks are cached under ``<save_dir>/ensemble/reward_net_{i}.pth``.

        Args:
            force_retrain: Retrain all networks even when cached files exist.

        Returns:
            List of trained :class:`TrajectoryRewardNet` (length ``n_ensemble``).
        """
        ensemble_dir = os.path.join(self.save_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)

        cache_paths = [
            os.path.join(ensemble_dir, f"reward_net_{i}.pth")
            for i in range(self.n_ensemble)
        ]
        all_cached = all(os.path.exists(p) for p in cache_paths)

        if all_cached and not force_retrain:
            self.reward_nets_ensemble = []
            for path in cache_paths:
                net = self._make_reward_net()
                net.load_state_dict(th.load(path, map_location=self.device))
                net.to(self.device)
                self.reward_nets_ensemble.append(net)
            return self.reward_nets_ensemble

        if not self.ranked_datasets:
            self._build_ranked_dataset()

        batch_size = kwargs.get("batch_size", 64)
        lr = kwargs.get("lr", 1e-4)
        weight_decay = kwargs.get("weight_decay", 0.01)
        n_steps = kwargs.get("n_steps", 1_000)

        def _collate(batch):
            return [b[0] for b in batch], [b[1] for b in batch]

        self.reward_nets_ensemble = []
        for i, dataset in enumerate(self.ranked_datasets):
            print(f"\n  [Ensemble {i + 1}/{self.n_ensemble}] Training reward network...")
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=_collate,
            )
            net = self._make_reward_net()
            learner = DemonstrationRankedIRL(
                reward_net=net,
                venv=self.venv,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                device=self.device,
            )
            learner.train(train_dataloader=dataloader, n_steps=n_steps)
            th.save(net.state_dict(), cache_paths[i])
            self.reward_nets_ensemble.append(net)

        return self.reward_nets_ensemble

    def _train_final_policy(self, total_timesteps: int, force_retrain: bool = False, **kwargs):
        """Train final policy via RL using the ensemble-averaged reward (Step 6).

        The reward signal is the mean prediction across all ``n_ensemble``
        networks, which reduces variance compared to any single network.

        Args:
            total_timesteps: Total environment steps for PPO.
            force_retrain: Retrain even when a cached policy exists.

        Returns:
            Trained :class:`PPO` agent.
        """
        final_policy_path = os.path.join(self.save_dir, "final_policy", "final_policy.zip")
        if os.path.exists(final_policy_path) and not force_retrain:
            self.final_policy = PPO.load(final_policy_path)
            return self.final_policy

        if not self.reward_nets_ensemble:
            self._train_reward_network()

        nets = self.reward_nets_ensemble

        def ensemble_reward_fn(obs, acts, next_obs, dones):
            return np.mean(
                [net.predict_processed(obs, acts, next_obs, dones, update_stats=False)
                 for net in nets],
                axis=0,
            )

        learned_reward_venv = RewardVecEnvWrapper(self.venv, ensemble_reward_fn)
        tb_log_dir = os.path.join(self.save_dir, "logs", "rl_step6")
        agent = PPO(
            "MlpPolicy",
            learned_reward_venv,
            n_steps=2048 // learned_reward_venv.num_envs,
            tensorboard_log=tb_log_dir,
        )
        agent.learn(total_timesteps=total_timesteps, progress_bar=True)

        final_policy_dir = os.path.join(self.save_dir, "final_policy")
        os.makedirs(final_policy_dir, exist_ok=True)
        agent.save(final_policy_path)
        self.final_policy = agent
        print(f"Final policy saved to {final_policy_path}")
        return self.final_policy

    def _make_reward_net(self) -> TrajectoryRewardNet:
        """Construct a fresh reward network with the standard architecture."""
        return TrajectoryRewardNet(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            use_state=True,
            use_action=False,
            use_next_state=False,
            use_done=False,
            hid_sizes=(256, 256),
        )
    
    def _solve_rtmpc(self, initial_state: np.ndarray, reference_trajectory: types.Trajectory) -> types.TrajectoryWithRew:

        current_noise_level = reference_trajectory.infos[0].get("noise_level")
        total_applied_noise_sum = reference_trajectory.infos[-1].get("total_applied_noise_sum")

        self.robust_mpc.mpc.set_initial_guess()
        self.robust_mpc.mpc.x0 = initial_state
        self.robust_mpc.simulator.x0 = initial_state
        
        state = initial_state

        builder = util.TrajectoryBuilder()
        builder.start_episode(initial_obs=initial_state)

        for t in range(len(reference_trajectory.obs)):
            # Compute nominal optimal control from MPC
            u_nom = self.robust_mpc.mpc.make_step(state)
            
            # Predict nominal state (first predicted state in the horizon)
            nominal_state = self.robust_mpc.mpc.data.prediction(("_x", "x"))[:, 0]

            # Convert to column vectors for disturbance rejection control law
            state_col = np.asarray(state).reshape(-1, 1)
            nominal_col = np.asarray(nominal_state).reshape(-1, 1)

            # Compute applied control with disturbance rejection control law
            applied_u = u_nom + self.robust_mpc.K @ (state_col - nominal_col)

            # Simulate the next state using the simulator
            next_state = self.robust_mpc.simulator.make_step(u0=applied_u)
            state = next_state

            builder.add_step(action=applied_u.flatten(), next_obs=next_state.flatten(), reward=0.0, info={"noise_level": current_noise_level, "total_applied_noise_sum": total_applied_noise_sum})

        rtmpc_trajectory = builder.finish()
    
        return rtmpc_trajectory
    
    @property
    def robust_mpc(self) -> RobustTubeMPC:
        """Return the current Robust Tube MPC."""
        return self._robust_mpc

    @robust_mpc.setter
    def robust_mpc(self, robust_mpc: RobustTubeMPC):
        """Function to catch external Robust Tube MPC. Main file running the trainer should set up the robust tube MPC and then pass it here."""
        self._robust_mpc = robust_mpc

    @property
    def policy(self) -> policies.BasePolicy:
        """Return the current policy."""
        if self.bc_policy is not None:
            return self.bc_policy
        if self.bc_trainer is not None and self.bc_trainer.policy is not None:
            return self.bc_trainer.policy
        raise ValueError("No policy available — call train() or _train_bc_policy() first.")

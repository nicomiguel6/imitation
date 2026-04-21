from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from stable_baselines3.common import policies
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import rollout, types
from imitation.scripts.NTRIL.noise_injection import EpsilonGreedyNoiseInjector
from imitation.scripts.SSRR.types import NoiseBucket


def generate_noisy_rollout_buckets(
    *,
    base_policy: policies.BasePolicy,
    venv: VecEnv,
    noise_levels: Sequence[float],
    n_rollouts_per_noise: int,
    rng: np.random.Generator,
    deterministic_policy: bool = False,
) -> Sequence[NoiseBucket]:
    """Collect noisy rollouts and group them by noise level.

    Notes:
    - We set `exclude_infos=False` so each trajectory retains `infos[0]["noise_level"]`,
      matching how NTRIL/D-REX code expects noise metadata to be stored.
    """
    buckets: List[NoiseBucket] = []
    injector = EpsilonGreedyNoiseInjector()
    for eta in noise_levels:
        noisy_policy = injector.inject_noise(base_policy, noise_level=float(eta))
        trajs = rollout.rollout(
            noisy_policy,
            venv,
            rollout.make_sample_until(min_episodes=int(n_rollouts_per_noise)),
            rng=rng,
            deterministic_policy=deterministic_policy,
            exclude_infos=False,
            label_info={"noise_level": float(eta)},
        )
        buckets.append(NoiseBucket(noise_level=float(eta), trajectories=trajs))
    return buckets


def buckets_to_dict(buckets: Sequence[NoiseBucket]) -> Dict[float, Sequence[types.TrajectoryWithRew]]:
    return {b.noise_level: b.trajectories for b in buckets}


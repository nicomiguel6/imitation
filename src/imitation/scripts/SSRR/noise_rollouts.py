from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import tqdm
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
    reference_trajectory: Optional[np.ndarray] = None,
) -> Sequence[NoiseBucket]:
    """Collect noisy rollouts and group them by noise level.

    Notes:
    - We set `exclude_infos=False` so each trajectory retains `infos[0]["noise_level"]`,
      matching how NTRIL/D-REX code expects noise metadata to be stored.
    """
    buckets: List[NoiseBucket] = []
    noise_injector = EpsilonGreedyNoiseInjector()
    for eta in tqdm.tqdm(noise_levels):
        noisy_policy = noise_injector.inject_noise(base_policy, noise_level=float(eta))
        rollouts = rollout.rollout(
                noisy_policy,
                venv,
                rollout.make_sample_until(
                    min_episodes=n_rollouts_per_noise,
                    min_timesteps=1000,
                ),
                rng=rng,
                label_info={"noise_level": float(eta)},
                exclude_infos=False,
                reset_options={"reference_trajectory": reference_trajectory}
                if reference_trajectory is not None
                else None,
            )
        buckets.append(NoiseBucket(noise_level=float(eta), trajectories=rollouts))
    
    return buckets


def buckets_to_dict(buckets: Sequence[NoiseBucket]) -> Dict[float, Sequence[types.TrajectoryWithRew]]:
    return {b.noise_level: b.trajectories for b in buckets}

def buckets_to_list(buckets: Sequence[NoiseBucket]) -> List[Sequence[types.TrajectoryWithRew]]:
    return [b.trajectories for b in buckets]


from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial import common
class EpsilonGreedyActionVecEnvWrapper(VecEnvWrapper):
    """Applies epsilon-greedy action noise at the VecEnv boundary.

    With probability `epsilon` (per-env, per-step), replaces the incoming action
    with a uniform random action in the action space bounds.

    This is a practical way to sample trajectories from the noisy policy
      pi_epsilon(a|s) = (1-epsilon) pi(a|s) + epsilon * U(a)
    while still using the underlying SB3 policy to compute log pi(a|s).
    """

    def __init__(self, venv: VecEnv, *, epsilon: float, rng: np.random.Generator):
        super().__init__(venv)
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = float(epsilon)
        self.rng = rng
        self.last_noise_mask: Optional[np.ndarray] = None

    def step_async(self, actions: np.ndarray) -> None:
        if self.epsilon <= 0.0:
            self.last_noise_mask = np.zeros(self.num_envs, dtype=bool)
            return self.venv.step_async(actions)

        mask = self.rng.random(self.num_envs) < self.epsilon
        self.last_noise_mask = mask
        if not mask.any():
            return self.venv.step_async(actions)

        noisy_actions = np.array(actions, copy=True)
        if isinstance(self.action_space, spaces.Discrete):
            noisy_actions[mask] = self.rng.integers(
                low=0,
                high=int(self.action_space.n),
                size=int(mask.sum()),
                dtype=np.int64,
            )
        elif isinstance(self.action_space, spaces.Box):
            low = np.asarray(self.action_space.low, dtype=np.float64)
            high = np.asarray(self.action_space.high, dtype=np.float64)
            # Broadcast to (n_noisy, act_dim)
            samp = self.rng.uniform(low=low, high=high, size=(int(mask.sum()),) + low.shape)
            noisy_actions[mask] = samp.astype(noisy_actions.dtype, copy=False)
        else:
            raise NotImplementedError(f"Unsupported action space {type(self.action_space)}")

        return self.venv.step_async(noisy_actions)

    def step_wait(self):
        return self.venv.step_wait()

    def reset(self, **kwargs):
        self.last_noise_mask = None
        return self.venv.reset(**kwargs)


def _log_uniform_prob(action_space: spaces.Space) -> float:
    if isinstance(action_space, spaces.Discrete):
        return -float(np.log(int(action_space.n)))
    if isinstance(action_space, spaces.Box):
        low = np.asarray(action_space.low, dtype=np.float64)
        high = np.asarray(action_space.high, dtype=np.float64)
        volume = float(np.prod(high - low))
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("Cannot compute uniform density for this Box action space")
        return -float(np.log(volume))
    raise NotImplementedError(f"Unsupported action space {type(action_space)}")


class NoisyAIRL(AIRL):
    """AIRL variant implementing Noisy-AIRL’s discriminator loss weighting (paper Eq. 6).

    This class assumes the generator trajectories are sampled from pi_eta (via
    an action-noise VecEnv wrapper). It then applies importance weights
    w = pi(a|s) / pi_eta(a|s) to the generator term in the discriminator loss.
    """

    def __init__(self, *args, noise_level: float, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0.0 <= noise_level <= 1.0):
            raise ValueError("noise_level must be in [0, 1]")
        self.noise_level = float(noise_level)
        self._log_u = _log_uniform_prob(self.venv.action_space)

    def _importance_weights(self, log_pi: th.Tensor) -> th.Tensor:
        """Compute w = pi / ((1-eps)*pi + eps*U) in log space for numerical stability."""
        eps = self.noise_level
        if eps <= 0.0:
            return th.ones_like(log_pi)

        # log pi_eta = log( (1-eps)*exp(log_pi) + eps*exp(log_u) )
        log_u = th.full_like(log_pi, fill_value=self._log_u)
        a = th.log(th.tensor(1.0 - eps, device=log_pi.device, dtype=log_pi.dtype)) + log_pi
        b = th.log(th.tensor(eps, device=log_pi.device, dtype=log_pi.dtype)) + log_u
        log_pi_eta = th.logaddexp(a, b)
        return th.exp(log_pi - log_pi_eta)

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        # Mostly mirrors common.AdversarialTrainer.train_disc, but with per-sample weights.
        with self.logger.accumulate_means("disc"):
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0
            self._disc_opt.zero_grad()

            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )
            last_loss = None
            last_logits = None
            last_labels = None
            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
                labels = batch["labels_expert_is_one"].float()

                # Importance weights apply only to generator samples (label==0).
                weights = th.ones_like(labels)
                logp = batch["log_policy_act_prob"]
                if logp is not None:
                    gen_mask = labels < 0.5
                    w = self._importance_weights(logp)
                    weights = th.where(gen_mask, w, weights)

                loss = F.binary_cross_entropy_with_logits(
                    disc_logits,
                    labels,
                    weight=weights,
                )

                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                loss *= self.demo_minibatch_size / self.demo_batch_size
                loss.backward()

                last_loss = loss
                last_logits = disc_logits
                last_labels = batch["labels_expert_is_one"]

            self._disc_opt.step()
            self._disc_step += 1

            with th.no_grad():
                assert last_loss is not None and last_logits is not None and last_labels is not None
                train_stats = common.compute_train_stats(
                    last_logits,
                    last_labels,
                    last_loss,
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", last_logits.detach())

        return train_stats


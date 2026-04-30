from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader

from imitation.data import types
from imitation.rewards import reward_nets

from imitation.scripts.SSRR.types import NoiseBucket, SigmoidParams, SnippetExample, SSRRRegressionConfig


class SnippetDataset(Dataset):
    """Samples contiguous snippets from noise-bucketed trajectories (SSRR Phase 3).

    Each example is (obs_snip, acts_snip, target_return).
    """

    def __init__(
        self,
        buckets: Sequence[NoiseBucket],
        sigmoid: SigmoidParams,
        *,
        num_snippets: int,
        cfg: SSRRRegressionConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        self.buckets = list(buckets)
        self.sigmoid = sigmoid
        self.num_snippets = int(num_snippets)
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()

        if self.cfg.min_steps < 1 or self.cfg.max_steps < 1:
            raise ValueError("min_steps/max_steps must be >= 1")
        if self.cfg.min_steps > self.cfg.max_steps:
            raise ValueError("min_steps must be <= max_steps")
        if not self.buckets:
            raise ValueError("Need at least one NoiseBucket")

    def __len__(self) -> int:
        return self.num_snippets

    def _sample_one(self) -> SnippetExample:
        b_idx = int(self.rng.integers(len(self.buckets)))
        bucket = self.buckets[b_idx]
        if not bucket.trajectories:
            raise ValueError(f"Noise bucket eta={bucket.noise_level} has no trajectories")
        t_idx = int(self.rng.integers(len(bucket.trajectories)))
        traj = bucket.trajectories[t_idx]

        T = len(traj.acts)
        if T < 1:
            raise ValueError("Trajectory must have at least 1 action")

        L = int(self.rng.integers(self.cfg.min_steps, self.cfg.max_steps + 1))
        L = min(L, T)
        start = int(self.rng.integers(0, T - L + 1))

        obs_snip = traj.obs[start : start + L + 1]
        acts_snip = traj.acts[start : start + L]

        base = float(self.sigmoid(bucket.noise_level))
        if self.cfg.length_normalize:
            # Matches paper appendix scaling: (L/T)*sigma(eta) plus a global constant scale.
            target = (base / float(T)) * float(L) * float(self.cfg.target_scale)
        else:
            target = base * float(self.cfg.target_scale)

        return SnippetExample(
            obs=np.asarray(obs_snip, dtype=np.float32),
            acts=np.asarray(acts_snip, dtype=np.float32),
            target_return=float(target),
            noise_level=float(bucket.noise_level),
        )

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        ex = self._sample_one()
        return ex.obs, ex.acts, ex.target_return


def _collate_snippets(
    batch: List[Tuple[np.ndarray, np.ndarray, float]],
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    # Variable-length snippets: keep as list-of-tensors; we’ll process in a loop.
    obs_list = [th.from_numpy(b[0]) for b in batch]   # (L+1, obs_dim)
    act_list = [th.from_numpy(b[1]) for b in batch]   # (L, act_dim)
    targets = th.tensor([b[2] for b in batch], dtype=th.float32)
    return obs_list, act_list, targets


class SSRRRegressor:
    """Fits R_theta by regressing snippet return to sigmoid target (paper Eq. 5)."""

    def __init__(
        self,
        reward_net: reward_nets.RewardNet,
        *,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        device: th.device | str = "cpu",
    ):
        self.reward_net = reward_net.to(device)
        self.device = th.device(device)
        self.optim = th.optim.Adam(self.reward_net.parameters(), lr=lr, weight_decay=weight_decay)

    def _snippet_return(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        # obs: (L+1, obs_dim), acts: (L, act_dim)
        cur = obs[:-1]
        nxt = obs[1:]
        done = th.zeros((acts.shape[0],), device=obs.device, dtype=th.float32)
        done[-1] = 1.0
        r = self.reward_net(cur, acts, nxt, done)
        return th.sum(r)

    def train(
        self,
        dataloader: DataLoader,
        *,
        n_steps: int,
        log_interval: int = 200,
    ) -> Dict[str, List[float]]:
        self.reward_net.train()
        losses: List[float] = []
        it = iter(dataloader)
        for step in range(1, n_steps + 1):
            try:
                obs_list, act_list, targets = next(it)
            except StopIteration:
                it = iter(dataloader)
                obs_list, act_list, targets = next(it)

            targets = targets.to(self.device)
            self.optim.zero_grad()

            preds = []
            for obs, acts in zip(obs_list, act_list):
                obs = obs.to(self.device).float()
                acts = acts.to(self.device).float()
                preds.append(self._snippet_return(obs, acts))
            pred_vec = th.stack(preds, dim=0)

            loss = th.mean((pred_vec - targets) ** 2)
            loss.backward()
            self.optim.step()

            losses.append(float(loss.item()))
        return {"step_losses": losses}


def make_dataloader(
    buckets: Sequence[NoiseBucket],
    sigmoid: SigmoidParams,
    *,
    num_snippets: int,
    cfg: SSRRRegressionConfig,
    batch_size: int,
    rng: Optional[np.random.Generator] = None,
) -> DataLoader:
    ds = SnippetDataset(buckets, sigmoid, num_snippets=num_snippets, cfg=cfg, rng=rng)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_snippets)


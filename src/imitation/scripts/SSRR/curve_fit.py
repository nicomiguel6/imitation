from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from imitation.data import types
from imitation.rewards import reward_nets

from imitation.scripts.SSRR.types import (
    CurveFitDiagnostics,
    NoiseBucket,
    NoisePerformanceData,
    SigmoidParams,
)


def _resize_to_unit_interval(arr: np.ndarray) -> np.ndarray:
    """Affine-rescale to [0, 1], matching the reference SSRR code behavior."""
    arr = np.asarray(arr, dtype=np.float64).copy()
    arr -= float(arr.min())
    maxv = float(arr.max())
    if maxv <= 0:
        return np.zeros_like(arr)
    arr *= 1.0 / maxv
    return arr


def estimate_airl_returns_by_noise(
    buckets: Sequence[NoiseBucket],
    airl_reward: reward_nets.RewardNet,
) -> NoisePerformanceData:
    """Compute AIRL-estimated cumulative returns per trajectory, grouped by noise.

    This corresponds to SSRR Phase 2’s y(eta) targets:
      y(eta) = mean_{tau ~ pi_eta} [ sum_t R_tilde(s_t, a_t) ]
    """
    noise_levels = []
    means = []
    stds = []
    returns_all = []

    for bucket in buckets:
        returns = []
        for traj in bucket.trajectories:
            obs = traj.obs
            acts = traj.acts
            next_obs = obs[1:]
            cur_obs = obs[:-1]
            done = np.zeros(len(acts), dtype=np.float32)
            done[-1] = float(traj.terminal)
            r = airl_reward.predict_processed(cur_obs, acts, next_obs, done, update_stats=False)
            returns.append(float(np.sum(r)))
        returns_all.append(returns)
        returns_arr = np.asarray(returns, dtype=np.float64)
        noise_levels.append(bucket.noise_level)
        means.append(float(np.mean(returns_arr)) if len(returns_arr) else 0.0)
        stds.append(float(np.std(returns_arr)) if len(returns_arr) else 0.0)

    return NoisePerformanceData(
        noise_levels=np.asarray(noise_levels, dtype=np.float64),
        returns_mean=np.asarray(means, dtype=np.float64),
        returns_std=np.asarray(stds, dtype=np.float64),
        returns_all=np.reshape(np.asarray(returns_all, dtype=np.float64), (-1,)),
    )


def _sigmoid(p: SigmoidParams | np.ndarray, x: np.ndarray) -> np.ndarray:
    x0, y0, c, k = p
    return c / (1.0 + np.exp(-k * (x - x0))) + y0


def fit_sigmoid_noise_performance_from_buckets(
    buckets: Sequence[NoiseBucket],
    airl_reward: reward_nets.RewardNet,
) -> Tuple[SigmoidParams, CurveFitDiagnostics]:
    """Fit SSRR’s 4-parameter sigmoid (paper Eq. 4) by least squares."""
    # Extract array of all noise levels and total rewards from each trajectory in each bucket
    noise_levels = []
    mean_rewards = []
    for bucket in buckets:
        total_rewards = []
        noise_levels.append(bucket[0].infos["noise_level"])
        for traj in bucket.trajectories:
            total_rewards.append(traj.infos["total_reward"])
        total_rewards = np.reshape(total_rewards, (-1,))
        mean_rewards.append(np.mean(total_rewards))
    
    data = NoisePerformanceData(noise_levels = noise_levels, returns_mean = mean_rewards)

    return fit_sigmoid_noise_performance(data, normalize_y=True, prefer_scipy=True)

def fit_sigmoid_noise_performance(
    data: NoisePerformanceData,
    *,
    normalize_y: bool = True,
    prefer_scipy: bool = True,
) -> Tuple[SigmoidParams, CurveFitDiagnostics]:
    """Fit SSRR’s 4-parameter sigmoid (paper Eq. 4) by least squares."""
    x = np.asarray(data.noise_levels, dtype=np.float64)
    y = np.asarray(data.returns_mean, dtype=np.float64)
    if normalize_y:
        y = _resize_to_unit_interval(y)

    # Reference impl uses (median(x), median(y), 1.0, -1.0) so slope decreases with noise.
    p0 = np.asarray([np.median(x), np.median(y), 1.0, -1.0], dtype=np.float64)

    p_opt: np.ndarray
    if prefer_scipy:
        try:
            from scipy.optimize import least_squares  # type: ignore

            def residuals(p: np.ndarray) -> np.ndarray:
                return y - _sigmoid(p, x)

            res = least_squares(residuals, p0, method="trf")
            p_opt = np.asarray(res.x, dtype=np.float64)
        except Exception:
            p_opt = p0
    else:
        p_opt = p0

    y_pred = _sigmoid(p_opt, x)
    resid = y - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    params = SigmoidParams(x0=float(p_opt[0]), y0=float(p_opt[1]), c=float(p_opt[2]), k=float(p_opt[3]))
    diag = CurveFitDiagnostics(
        r2=float(r2),
        residuals=resid,
        y_true=y,
        y_pred=y_pred,
    )
    return params, diag


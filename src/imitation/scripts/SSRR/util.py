"""Visualization helpers for the SSRR reward regression pipeline.

Provides reusable functions for plotting the mean learned reward surface
over the (position, velocity) state space — analogous to
``imitation.scripts.NTRIL.double_integrator.util.plot_learned_reward_network``.

Author: Nicolas Miguel
Date: May 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import gymnasium as gym

from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


# ---------------------------------------------------------------------------
# Learned reward surface
# ---------------------------------------------------------------------------

def plot_learned_reward(
    ensemble: Sequence[BasicRewardNet],
    observation_space: gym.Space,
    action_space: gym.Space,
    *,
    ref_pos: float = 0.0,
    ref_vel: float = 0.0,
    n_grid: int = 100,
    device: str = "cpu",
    reference_trajectory: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the mean learned reward contour over the (position, velocity) grid.

    Sweeps ``pos`` and ``vel`` on a uniform meshgrid while holding
    ``ref_pos`` and ``ref_vel`` constant (i.e. a fixed tracking target).
    Each ensemble member's per-step prediction is averaged to obtain the
    mean reward surface.

    Args:
        ensemble: Trained ``BasicRewardNet`` ensemble (already on *device*).
        observation_space: Environment observation space (used for axis bounds).
        action_space: Environment action space (used to size the dummy action).
        ref_pos: Constant reference position appended to every grid state.
        ref_vel: Constant reference velocity appended to every grid state.
        n_grid: Number of grid points along each axis.
        device: Torch device string.
        reference_trajectory: Optional ``(T, obs_dim)`` array whose first two
            columns are position and velocity; overlaid on the contour when given.
        out_path: If provided, the figure is saved here and closed.
        title: Plot title.  Defaults to ``"Mean learned reward (SSRR ensemble)"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    obs_low = observation_space.low
    obs_high = observation_space.high

    pos_vals = np.linspace(obs_low[0], obs_high[0], n_grid)
    vel_vals = np.linspace(obs_low[1], obs_high[1], n_grid)
    pos_grid, vel_grid = np.meshgrid(pos_vals, vel_vals)

    n_pts = n_grid * n_grid
    obs_flat = np.stack(
        [
            pos_grid.ravel(),
            vel_grid.ravel(),
            np.full(n_pts, ref_pos, dtype=np.float32),
            np.full(n_pts, ref_vel, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    action_dim = action_space.shape[0]
    dummy_acts = np.zeros((n_pts, action_dim), dtype=np.float32)
    dummy_next = np.zeros_like(obs_flat)
    dummy_done = np.zeros(n_pts, dtype=np.float32)

    per_net = np.stack(
        [
            net.predict_processed(
                obs_flat, dummy_acts, dummy_next, dummy_done, update_stats=False
            )
            for net in ensemble
        ],
        axis=0,
    )
    reward_grid = per_net.mean(axis=0).reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(pos_grid, vel_grid, reward_grid, levels=50, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="R_θ(s)  [ensemble mean]")

    if reference_trajectory is not None:
        traj_pos = reference_trajectory[:, 0]
        traj_vel = reference_trajectory[:, 1]
        ax.plot(traj_pos, traj_vel, "w-", linewidth=1.5, label="Reference trajectory")
        ax.plot(traj_pos[0], traj_vel[0], "wo", markersize=7, label="Start")
        ax.plot(traj_pos[-1], traj_vel[-1], "w*", markersize=10, label="End")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title or "Mean learned reward (SSRR ensemble)")
    ax.grid(True, alpha=0.2)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    return fig, ax


def load_ssrr_ensemble(
    ensemble_dir: Path,
    observation_space: gym.Space,
    action_space: gym.Space,
    n_ensemble: int = 3,
    device: str = "cpu",
) -> List[BasicRewardNet]:
    """Load a saved SSRR reward-network ensemble from *ensemble_dir*.

    Expects files named ``ssrr_reward_0.pt``, ``ssrr_reward_1.pt``, … inside
    *ensemble_dir* (the naming convention used by
    ``test_reward_regression_step.py``).

    Args:
        ensemble_dir: Directory containing the saved ``.pt`` state-dicts.
        observation_space: Environment observation space.
        action_space: Environment action space.
        n_ensemble: Number of ensemble members to load.
        device: Torch device string.

    Returns:
        List of loaded ``BasicRewardNet`` instances in eval mode.
    """
    nets: List[BasicRewardNet] = []
    for i in range(n_ensemble):
        path = ensemble_dir / f"ssrr_reward_{i}.pt"
        net = BasicRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            normalize_input_layer=RunningNorm,
        )
        net.load_state_dict(th.load(str(path), map_location=device))
        net.to(device)
        net.eval()
        nets.append(net)
    return nets

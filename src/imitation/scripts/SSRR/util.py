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
from matplotlib import cm, colors
from matplotlib.widgets import Slider
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


def plot_learned_reward_time(
    ensemble: Sequence[BasicRewardNet],
    observation_space: gym.Space,
    action_space: gym.Space,
    *,
    reference_trajectory: np.ndarray,
    n_grid: int = 80,
    max_slices: int = 50,
    device: str = "cpu",
    out_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot time-indexed learned-reward contours as a 3D volume.

    For each selected time-step ``t`` in ``reference_trajectory``, this function
    evaluates the ensemble reward over a ``(position, velocity)`` mesh while
    fixing the reference state ``(ref_pos_t, ref_vel_t)``. Each reward slice is
    rendered as a 2D contour in 3D space at ``y=t``.

    The resulting axes are:
      - x-axis: position
      - y-axis: time-step index
      - z-axis: velocity
    """
    if reference_trajectory.ndim != 2 or reference_trajectory.shape[1] < 2:
        raise ValueError(
            "reference_trajectory must be shaped (T, obs_dim) with at least 2 columns."
        )

    obs_low = observation_space.low
    obs_high = observation_space.high
    pos_vals = np.linspace(obs_low[0], obs_high[0], n_grid)
    vel_vals = np.linspace(obs_low[1], obs_high[1], n_grid)
    pos_grid, vel_grid = np.meshgrid(pos_vals, vel_vals)
    n_pts = n_grid * n_grid

    action_dim = action_space.shape[0]
    dummy_acts = np.zeros((n_pts, action_dim), dtype=np.float32)
    dummy_done = np.zeros(n_pts, dtype=np.float32)

    t_count = reference_trajectory.shape[0]
    n_slices = min(max_slices, t_count)
    t_indices = np.linspace(0, t_count - 1, n_slices, dtype=int)

    reward_slices = []
    for t in t_indices:
        ref_pos_t = float(reference_trajectory[t, 0])
        ref_vel_t = float(reference_trajectory[t, 1])
        obs_flat = np.stack(
            [
                pos_grid.ravel(),
                vel_grid.ravel(),
                np.full(n_pts, ref_pos_t, dtype=np.float32),
                np.full(n_pts, ref_vel_t, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        dummy_next = np.zeros_like(obs_flat)

        per_net = np.stack(
            [
                net.predict_processed(
                    obs_flat,
                    dummy_acts,
                    dummy_next,
                    dummy_done,
                    update_stats=False,
                )
                for net in ensemble
            ],
            axis=0,
        )
        reward_slices.append(per_net.mean(axis=0).reshape(n_grid, n_grid))

    reward_stack = np.stack(reward_slices, axis=0)
    vmin = float(np.min(reward_stack))
    vmax = float(np.max(reward_stack))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("viridis")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for time_idx, reward_grid in zip(t_indices, reward_slices):
        face_colors = cmap(norm(reward_grid))
        ax.contourf(
            pos_grid,
            vel_grid,
            reward_grid,
            zdir="y",
            offset=float(time_idx),
            levels=30,
            cmap=cmap,
            norm=norm,
            alpha=0.8,
        )
        # Add a light wireframe to improve slice readability in 3D perspective.
        ax.plot_surface(
            pos_grid,
            np.full_like(pos_grid, float(time_idx)),
            vel_grid,
            facecolors=face_colors,
            rstride=max(1, n_grid // 25),
            cstride=max(1, n_grid // 25),
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.08,
        )

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.1, label="R_θ(s)  [ensemble mean]")

    ax.set_xlabel("Position")
    ax.set_ylabel("Time step")
    ax.set_zlabel("Velocity")
    ax.set_xlim(float(obs_low[0]), float(obs_high[0]))
    ax.set_ylim(float(t_indices[0]), float(t_indices[-1]))
    ax.set_zlim(float(obs_low[1]), float(obs_high[1]))
    ax.set_title(title or "Time-varying mean learned reward (SSRR ensemble)")
    ax.view_init(elev=25, azim=-60)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    return fig, ax


def plot_learned_reward_time_slider(
    ensemble: Sequence[BasicRewardNet],
    observation_space: gym.Space,
    action_space: gym.Space,
    *,
    reference_trajectory: np.ndarray,
    n_grid: int = 100,
    max_frames: int = 250,
    device: str = "cpu",
    out_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, Slider]:
    """Plot a 2D contour with an interactive time slider.

    Precomputes reward contours over selected reference time steps and provides
    a slider to scrub through frames while keeping a fixed color scale.
    """
    del device  # Kept for API consistency with other plotting helpers.

    if reference_trajectory.ndim != 2 or reference_trajectory.shape[1] < 2:
        raise ValueError(
            "reference_trajectory must be shaped (T, obs_dim) with at least 2 columns."
        )

    obs_low = observation_space.low
    obs_high = observation_space.high
    pos_vals = np.linspace(obs_low[0], obs_high[0], n_grid)
    vel_vals = np.linspace(obs_low[1], obs_high[1], n_grid)
    pos_grid, vel_grid = np.meshgrid(pos_vals, vel_vals)
    n_pts = n_grid * n_grid

    action_dim = action_space.shape[0]
    dummy_acts = np.zeros((n_pts, action_dim), dtype=np.float32)
    dummy_done = np.zeros(n_pts, dtype=np.float32)

    t_count = reference_trajectory.shape[0]
    n_frames = min(max_frames, t_count)
    t_indices = np.linspace(0, t_count - 1, n_frames, dtype=int)

    reward_slices = []
    for t in t_indices:
        ref_pos_t = float(reference_trajectory[t, 0])
        ref_vel_t = float(reference_trajectory[t, 1])
        obs_flat = np.stack(
            [
                pos_grid.ravel(),
                vel_grid.ravel(),
                np.full(n_pts, ref_pos_t, dtype=np.float32),
                np.full(n_pts, ref_vel_t, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        dummy_next = np.zeros_like(obs_flat)

        per_net = np.stack(
            [
                net.predict_processed(
                    obs_flat,
                    dummy_acts,
                    dummy_next,
                    dummy_done,
                    update_stats=False,
                )
                for net in ensemble
            ],
            axis=0,
        )
        reward_slices.append(per_net.mean(axis=0).reshape(n_grid, n_grid))

    reward_stack = np.stack(reward_slices, axis=0)
    norm = colors.Normalize(
        vmin=float(np.min(reward_stack)),
        vmax=float(np.max(reward_stack)),
    )
    cmap = cm.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.18)

    traj_pos = reference_trajectory[t_indices, 0]
    traj_vel = reference_trajectory[t_indices, 1]
    ax.plot(traj_pos, traj_vel, "w--", linewidth=1.0, alpha=0.9, label="Reference path")
    (marker,) = ax.plot([traj_pos[0]], [traj_vel[0]], "wo", markersize=7, label="Current ref")

    contour_set = ax.contourf(
        pos_grid,
        vel_grid,
        reward_slices[0],
        levels=50,
        cmap=cmap,
        norm=norm,
    )

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, label="R_θ(s)  [ensemble mean]")

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_xlim(float(obs_low[0]), float(obs_high[0]))
    ax.set_ylim(float(obs_low[1]), float(obs_high[1]))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    def _set_title(frame_idx: int) -> None:
        t = int(t_indices[frame_idx])
        ref_pos_t = float(reference_trajectory[t, 0])
        ref_vel_t = float(reference_trajectory[t, 1])
        base_title = title or "Time-varying mean learned reward (slider view)"
        ax.set_title(
            f"{base_title}\nframe={frame_idx + 1}/{n_frames}, t={t}, "
            f"ref=({ref_pos_t:.3f}, {ref_vel_t:.3f})"
        )

    _set_title(0)

    slider_ax = fig.add_axes([0.18, 0.07, 0.64, 0.035])
    slider = Slider(
        ax=slider_ax,
        label="Frame",
        valmin=0,
        valmax=n_frames - 1,
        valinit=0,
        valstep=1,
    )

    def _update(_val: float) -> None:
        nonlocal contour_set
        frame_idx = int(slider.val)
        for coll in contour_set.collections:
            coll.remove()
        contour_set = ax.contourf(
            pos_grid,
            vel_grid,
            reward_slices[frame_idx],
            levels=50,
            cmap=cmap,
            norm=norm,
        )
        marker.set_data([traj_pos[frame_idx]], [traj_vel[frame_idx]])
        _set_title(frame_idx)
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig, ax, slider


def reward_sanity_gate(
    ensemble: Sequence[BasicRewardNet],
    observation_space: gym.Space,
    action_space: gym.Space,
    reference_trajectory: np.ndarray,
    *,
    n_grid: int = 80,
    n_frames: int = 50,
    device: str = "cpu",
    argmax_dist_tol: float = 0.5,
    on_ref_percentile_thresh: float = 0.75,
    raise_on_fail: bool = False,
    verbose: bool = True,
) -> dict:
    """Sanity-check a learned reward *before* spending compute on RL.

    The check directly tests whether the ensemble-mean reward is oriented toward
    the reference trajectory. For a sample of reference frames it sweeps the
    (position, velocity) grid (holding ref fixed) and measures:

    * ``argmax_dist`` — normalized distance between the grid argmax and the true
      reference state. ~0 means the reward peaks at the reference (good).
    * ``on_ref_percentile`` — fraction of grid points whose reward is <= the
      reward evaluated *at* the reference. ~1 means the reference is (near) the
      global max (good).
    * ``flatness`` — reward std divided by |mean| over the grid; tiny values flag
      a near-constant reward that gives RL no usable gradient.

    The gate passes when the median ``argmax_dist`` <= ``argmax_dist_tol`` AND the
    median ``on_ref_percentile`` >= ``on_ref_percentile_thresh``.

    Args:
        ensemble: Trained reward nets (averaged via ``predict_processed``).
        observation_space: Used for grid bounds. Assumes obs = [pos, vel, ref_pos, ref_vel].
        action_space: Used to size the (zero) dummy action.
        reference_trajectory: ``(T, >=2)`` array; columns 0,1 are pos,vel.
        n_grid: Grid resolution per axis.
        n_frames: Number of reference frames to sample across the trajectory.
        device: Torch device string (unused beyond net placement).
        argmax_dist_tol: Max allowed median normalized argmax distance.
        on_ref_percentile_thresh: Min allowed median on-reference percentile.
        raise_on_fail: If True, raise ``RuntimeError`` when the gate fails.
        verbose: Print a human-readable summary.

    Returns:
        Dict of aggregate metrics plus a boolean ``"pass"``.
    """
    ref = np.asarray(reference_trajectory, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] < 2:
        raise ValueError("reference_trajectory must be (T, >=2): [pos, vel, ...]")

    obs_low = observation_space.low
    obs_high = observation_space.high
    pos_vals = np.linspace(obs_low[0], obs_high[0], n_grid)
    vel_vals = np.linspace(obs_low[1], obs_high[1], n_grid)
    pos_grid, vel_grid = np.meshgrid(pos_vals, vel_vals)
    n_pts = n_grid * n_grid
    pos_half = 0.5 * float(obs_high[0] - obs_low[0])
    vel_half = 0.5 * float(obs_high[1] - obs_low[1])

    action_dim = action_space.shape[0]
    dummy_acts = np.zeros((n_pts + 1, action_dim), dtype=np.float32)
    dummy_done = np.zeros(n_pts + 1, dtype=np.float32)

    t_count = ref.shape[0]
    n_frames = min(n_frames, t_count)
    t_indices = np.linspace(0, t_count - 1, n_frames, dtype=int)

    argmax_dists: List[float] = []
    on_ref_percentiles: List[float] = []
    flatness_vals: List[float] = []
    grid_means: List[float] = []
    grid_stds: List[float] = []

    for t in t_indices:
        ref_pos_t = float(ref[t, 0])
        ref_vel_t = float(ref[t, 1])
        # Grid states + the exact reference state appended as the final row.
        obs_flat = np.stack(
            [
                np.concatenate([pos_grid.ravel(), [ref_pos_t]]),
                np.concatenate([vel_grid.ravel(), [ref_vel_t]]),
                np.full(n_pts + 1, ref_pos_t),
                np.full(n_pts + 1, ref_vel_t),
            ],
            axis=1,
        ).astype(np.float32)
        dummy_next = np.zeros_like(obs_flat)

        per_net = np.stack(
            [
                net.predict_processed(
                    obs_flat, dummy_acts, dummy_next, dummy_done, update_stats=False
                )
                for net in ensemble
            ],
            axis=0,
        )
        r = per_net.mean(axis=0)
        grid_r = r[:-1]
        on_ref_r = float(r[-1])

        amax = int(np.argmax(grid_r))
        pa = float(pos_grid.ravel()[amax])
        va = float(vel_grid.ravel()[amax])
        dist = np.sqrt(
            ((pa - ref_pos_t) / max(pos_half, 1e-9)) ** 2
            + ((va - ref_vel_t) / max(vel_half, 1e-9)) ** 2
        )
        argmax_dists.append(float(dist))
        on_ref_percentiles.append(float(np.mean(grid_r <= on_ref_r)))
        flatness_vals.append(float(np.std(grid_r) / (abs(np.mean(grid_r)) + 1e-12)))
        grid_means.append(float(np.mean(grid_r)))
        grid_stds.append(float(np.std(grid_r)))

    med_dist = float(np.median(argmax_dists))
    med_pct = float(np.median(on_ref_percentiles))
    med_flat = float(np.median(flatness_vals))
    passed = (med_dist <= argmax_dist_tol) and (med_pct >= on_ref_percentile_thresh)

    result = {
        "pass": bool(passed),
        "median_argmax_dist": med_dist,
        "median_on_ref_percentile": med_pct,
        "median_flatness": med_flat,
        "frac_frames_argmax_ok": float(np.mean(np.asarray(argmax_dists) <= argmax_dist_tol)),
        "frac_frames_ref_is_top": float(np.mean(np.asarray(on_ref_percentiles) >= on_ref_percentile_thresh)),
        "argmax_dist_tol": argmax_dist_tol,
        "on_ref_percentile_thresh": on_ref_percentile_thresh,
        # Reward-scale stats (averaged over sampled frames) — useful for picking an
        # RL-side affine rescale (a*(r-b)) that does not change the optimal policy.
        "reward_grid_mean": float(np.mean(grid_means)),
        "reward_grid_std": float(np.mean(grid_stds)),
    }

    if verbose:
        status = "PASS" if passed else "FAIL"
        print("\n=== Reward sanity gate ===")
        print(f"  median argmax distance to reference = {med_dist:.3f}  (tol <= {argmax_dist_tol})")
        print(f"  median on-reference percentile      = {med_pct:.3f}  (need >= {on_ref_percentile_thresh})")
        print(f"  median flatness (std/|mean|)        = {med_flat:.4g}  (tiny => near-constant reward)")
        print(f"  frames argmax near ref: {result['frac_frames_argmax_ok']*100:.0f}%  | "
              f"frames ref is top-{int((1-on_ref_percentile_thresh)*100)}%: {result['frac_frames_ref_is_top']*100:.0f}%")
        print(f"  --> {status}")
        if not passed:
            print("  Reward does NOT favor the reference. RL on this reward will likely fail.")

    if not passed and raise_on_fail:
        raise RuntimeError(
            f"Reward sanity gate FAILED: median_argmax_dist={med_dist:.3f} "
            f"(tol {argmax_dist_tol}), median_on_ref_percentile={med_pct:.3f} "
            f"(thresh {on_ref_percentile_thresh})."
        )

    return result


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
    if n_ensemble == 1:
        # Load the only possible file from the directory, whatever its name is
        files = list(ensemble_dir.glob("*.pt"))
        if len(files) != 1:
            raise FileNotFoundError(f"Expected exactly 1 .pt file in {ensemble_dir}, found {len(files)}")
        path = files[0]
 
        net = BasicRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            normalize_input_layer=RunningNorm,
        )
        net.load_state_dict(th.load(str(path), map_location=device))
        net.to(device)
        net.eval()
        nets.append(net)
    else:
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

if __name__ == "__main__":
    # Set up dummy env
    env = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0")
    observation_space = env.observation_space
    action_space = env.action_space
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    reference_trajectory = np.load("/home/nicomiguel/imitation/src/imitation/scripts/SSRR/tests/airl_outputs/20260420_231943_sinusoidal_A1.0_f0.01/reference_trajectory.npy")
    
    # load ensemble of reward net
    ensemble = load_ssrr_ensemble(
        # ensemble_dir=Path("/home/nicomiguel/imitation/src/imitation/scripts/SSRR/tests/airl_outputs/20260420_231943_sinusoidal_A1.0_f0.01/best_airl"),
        ensemble_dir=Path("/home/nicomiguel/imitation/src/imitation/scripts/SSRR/tests/airl_outputs/20260527_231943_sinusoidal_A1.0_f0.01/ssrr_regression/ensemble"),
        observation_space=observation_space,
        action_space=action_space,
        n_ensemble=3,
        device=device,
    )

    # plot reward time slider
    fig, ax, slider = plot_learned_reward_time_slider(
        ensemble=ensemble,
        observation_space=observation_space,
        action_space=action_space,
        reference_trajectory=reference_trajectory,
        n_grid=100,
        max_frames=250,
        device=device,
    )
    plt.show()
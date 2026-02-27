"""
Debug and visualization helpers for the double integrator example.

Provides reusable functions for evaluating trained policies, inspecting
augmented trajectory snippets, simulating the ancillary controller, and
plotting results — all without running the full main_sim.py pipeline.

Author: Nicolas Miguel
Date: January 2026
"""

from __future__ import annotations

import argparse
import os  # noqa: F401
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch as th  # noqa: F401
import gymnasium as gym  # noqa: F401
from scipy.linalg import solve_discrete_are
from stable_baselines3 import PPO  # noqa: F401

from imitation.algorithms import bc  # noqa: F401
from imitation.data import rollout, serialize, types  # noqa: F401
from imitation.data.wrappers import RolloutInfoWrapper  # noqa: F401
from imitation.util import util, logger as imit_logger  # noqa: F401
from imitation.scripts.NTRIL.ntril import NTRILTrainer  # noqa: F401
from imitation.rewards.reward_nets import BasicRewardNet  # noqa: F401
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper  # noqa: F401
from imitation.scripts.NTRIL.noise_injection import EpsilonGreedyNoiseInjector  # noqa: F401
from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC, RobustTubeMPCPolicy  # noqa: F401
from imitation.scripts.NTRIL.testcode.apply_robust_tube_mpc import main as apply_robust_tube_mpc  # noqa: F401
from imitation.rewards.reward_nets import TrajectoryRewardNet  # noqa: F401

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_SAVE_DIR = SCRIPT_DIR / "ntril_outputs"


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def test_trained_policy(policy, env, n_episodes=10):
    """Test a trained policy on a given environment."""
    returns = []
    lengths = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            action = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1
        returns.append(episode_return)
        lengths.append(episode_length)
    return returns, lengths


# ---------------------------------------------------------------------------
# General plotting
# ---------------------------------------------------------------------------

def plot_phase_portrait(states, title) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the phase portrait of given states."""
    fig, ax = plt.subplots()
    ax.plot(states[:, 0], states[:, 1], "k-", label="trajectory")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    ax.grid(True)
    return fig, ax


# ---------------------------------------------------------------------------
# Data loading (augmented snippets / RTMPC trajectories)
# ---------------------------------------------------------------------------

def load_augmented_data(
    save_dir: str | Path = DEFAULT_SAVE_DIR,
    noise_level: float = 0.0,
) -> Tuple[List[types.Trajectory], List[types.TrajectoryWithRew]]:
    """Load RTMPC nominal trajectories and augmented snippets for one noise level.

    Args:
        save_dir: Root output directory produced by NTRILTrainer.
        noise_level: Which noise bin to load.

    Returns:
        Tuple of (rtmpc_trajectories, augmented_snippets).
    """
    save_dir = Path(save_dir)
    tag = f"noise_{noise_level:.2f}"

    rtmpc_path = save_dir / "rtmpc_trajectories" / f"{tag}.pkl"
    aug_path = save_dir / "augmented_data" / f"{tag}.pkl"

    if not rtmpc_path.exists():
        raise FileNotFoundError(f"RTMPC trajectories not found at {rtmpc_path}")
    if not aug_path.exists():
        raise FileNotFoundError(f"Augmented data not found at {aug_path}")

    rtmpc_trajs = serialize.load(str(rtmpc_path))
    augmented_snippets = serialize.load(str(aug_path))

    print(f"Loaded {len(rtmpc_trajs)} RTMPC trajectories and "
          f"{len(augmented_snippets)} augmented snippets for noise={noise_level:.2f}")
    return rtmpc_trajs, augmented_snippets


# ---------------------------------------------------------------------------
# Ancillary-controller forward simulation
# ---------------------------------------------------------------------------

def default_double_integrator_params() -> dict:
    """Return the standard double-integrator dynamics and cost matrices.

    Continuous-time A, B are discretised at dt=1.0 via Euler (matching do_mpc
    collocation at the same t_step for the double integrator).
    """
    A_c = np.array([[0.0, 1.0], [0.0, 0.0]])
    B_c = np.array([[0.0], [1.0]])
    dt = 1.0
    A_d = np.eye(2) + A_c * dt
    B_d = B_c * dt
    Q = np.diag([10.0, 1.0])
    R = 0.01 * np.eye(1)
    return dict(A_d=A_d, B_d=B_d, Q=Q, R=R)


def build_ancillary_gain(
    A_d: np.ndarray,
    B_d: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Compute the LQR gain K = -inv(B'PB + R)(B'PA) from the DARE solution."""
    P = solve_discrete_are(A_d, B_d, Q, R)
    return -np.linalg.inv(B_d.T @ P @ B_d + R) @ (B_d.T @ P @ A_d)


def simulate_ancillary_controller(
    initial_state: np.ndarray,
    reference_obs: np.ndarray,
    reference_acts: np.ndarray,
    A_d: np.ndarray,
    B_d: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the ancillary controller u = u0 + K(x - x0) forward in time.

    Uses discrete-time dynamics x_{k+1} = A_d x_k + B_d u_k.

    Args:
        initial_state: Starting state for the simulation.
        reference_obs: Nominal observation sequence to track (T+1, state_dim).
        reference_acts: Nominal action sequence (T, action_dim).
        A_d: Discrete-time state matrix.
        B_d: Discrete-time input matrix.
        K: Ancillary gain matrix (from DARE).

    Returns:
        Tuple of (states, actions) arrays, shapes (T+1, n) and (T, m).
    """
    T = len(reference_acts)
    state_dim = A_d.shape[0]
    action_dim = B_d.shape[1]

    states = np.zeros((T + 1, state_dim))
    actions = np.zeros((T, action_dim))

    states[0] = initial_state
    x = initial_state.reshape(-1, 1)

    for t in range(T):
        x0 = reference_obs[t].reshape(-1, 1)
        u0 = reference_acts[t].reshape(-1, 1)
        u = u0 + K @ (x - x0)
        x = A_d @ x + B_d @ u
        states[t + 1] = x.flatten()
        actions[t] = u.flatten()

    return states, actions


# ---------------------------------------------------------------------------
# Snippet inspection plots
# ---------------------------------------------------------------------------

def _find_best_reference_offset(
    snippet: types.TrajectoryWithRew,
    rtmpc_traj: types.Trajectory,
) -> int:
    """Heuristic: find the time offset in rtmpc_traj whose state is closest
    to the snippet's initial observation."""
    dists = np.linalg.norm(rtmpc_traj.obs - snippet.obs[0], axis=1)
    return int(np.argmin(dists))


def plot_snippets(
    rtmpc_traj: types.Trajectory,
    snippets: Sequence[types.TrajectoryWithRew],
    state_idx: int = 0,
    title: Optional[str] = None,
    simulate: bool = False,
    dynamics_params: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot augmented snippets overlaid on the RTMPC nominal trajectory.

    Args:
        rtmpc_traj: Nominal RTMPC trajectory (the reference).
        snippets: Augmented trajectory snippets to overlay.
        state_idx: Which state dimension to plot on the y-axis.
        title: Plot title.
        simulate: If True, re-simulate each snippet with the ancillary
            controller instead of using the stored observations.
        dynamics_params: Dict with keys {A_d, B_d, Q, R}. Required when
            *simulate* is True; ignored otherwise.  Uses double-integrator
            defaults when None.
        ax: Optional existing axes to draw on.

    Returns:
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    state_labels = {0: "Position", 1: "Velocity"}
    ylabel = state_labels.get(state_idx, f"State[{state_idx}]")

    ax.plot(rtmpc_traj.obs[:, state_idx], "b-", linewidth=2, label="RTMPC nominal")

    if simulate:
        params = dynamics_params or default_double_integrator_params()
        K = build_ancillary_gain(params["A_d"], params["B_d"], params["Q"], params["R"])

    for i, snip in enumerate(snippets):
        t_start = _find_best_reference_offset(snip, rtmpc_traj)
        t_range = np.arange(t_start, t_start + len(snip.obs))

        if simulate:
            horizon = min(len(snip.acts), len(rtmpc_traj.acts) - t_start)
            if horizon < 1:
                continue
            ref_obs = rtmpc_traj.obs[t_start : t_start + horizon + 1]
            ref_acts = rtmpc_traj.acts[t_start : t_start + horizon]
            sim_states, _ = simulate_ancillary_controller(
                snip.obs[0], ref_obs, ref_acts,
                params["A_d"], params["B_d"], K,
            )
            t_range = np.arange(t_start, t_start + len(sim_states))
            y = sim_states[:, state_idx]
        else:
            y = snip.obs[:, state_idx]

        label = "Augmented snippet" if i == 0 else None
        ax.plot(t_range, y, "r-", alpha=0.5, linewidth=0.8, label=label)

    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Augmented snippets vs RTMPC nominal trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_snippet_phase(
    rtmpc_traj: types.Trajectory,
    snippets: Sequence[types.TrajectoryWithRew],
    title: Optional[str] = None,
    simulate: bool = False,
    dynamics_params: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Phase-portrait (position vs velocity) of snippets and RTMPC trajectory.

    Args:
        rtmpc_traj: Nominal RTMPC trajectory.
        snippets: Augmented snippets.
        title: Plot title.
        simulate: Re-simulate with the ancillary controller.
        dynamics_params: Dynamics / cost matrices (see *plot_snippets*).
        ax: Existing axes.

    Returns:
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    ax.plot(
        rtmpc_traj.obs[:, 0], rtmpc_traj.obs[:, 1],
        "b-", linewidth=2, label="RTMPC nominal",
    )

    if simulate:
        params = dynamics_params or default_double_integrator_params()
        K = build_ancillary_gain(params["A_d"], params["B_d"], params["Q"], params["R"])

    for i, snip in enumerate(snippets):
        if simulate:
            t_start = _find_best_reference_offset(snip, rtmpc_traj)
            horizon = min(len(snip.acts), len(rtmpc_traj.acts) - t_start)
            if horizon < 1:
                continue
            ref_obs = rtmpc_traj.obs[t_start : t_start + horizon + 1]
            ref_acts = rtmpc_traj.acts[t_start : t_start + horizon]
            sim_states, _ = simulate_ancillary_controller(
                snip.obs[0], ref_obs, ref_acts,
                params["A_d"], params["B_d"], K,
            )
            px, vx = sim_states[:, 0], sim_states[:, 1]
        else:
            px, vx = snip.obs[:, 0], snip.obs[:, 1]

        label = "Augmented snippet" if i == 0 else None
        ax.plot(px, vx, "r-", alpha=0.5, linewidth=0.8, label=label)
        ax.plot(px[0], vx[0], "go", markersize=4, zorder=5)

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title or "Phase portrait: snippets vs RTMPC nominal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    return fig, ax


def inspect_snippets(
    save_dir: str | Path = DEFAULT_SAVE_DIR,
    noise_level: float = 0.0,
    traj_idx: int = 0,
    n_snippets: int = 10,
    simulate: bool = False,
    state_idx: int = 0,
    out_dir: Optional[str | Path] = None,
) -> None:
    """High-level convenience: load augmented snippets, plot, and save figures.

    Args:
        save_dir: NTRILTrainer output directory.
        noise_level: Noise bin to inspect.
        traj_idx: Which RTMPC base trajectory to use as reference.
        n_snippets: Max number of snippets to overlay.
        simulate: Re-simulate snippets with the ancillary controller.
        state_idx: State dimension for the time-series plot.
        out_dir: Directory for saved figures. Defaults to
            ``<save_dir>/debug/plots/snippet_inspection``.
    """
    rtmpc_trajs, snippets = load_augmented_data(save_dir, noise_level)

    if traj_idx >= len(rtmpc_trajs):
        raise IndexError(
            f"traj_idx={traj_idx} but only {len(rtmpc_trajs)} RTMPC trajectories exist"
        )

    rtmpc_ref = rtmpc_trajs[traj_idx]
    subset = snippets[:n_snippets]

    tag = f"noise_{noise_level:.2f}_traj_{traj_idx}"
    sim_tag = "_simulated" if simulate else ""

    fig_ts, _ = plot_snippets(
        rtmpc_ref, subset,
        state_idx=state_idx,
        simulate=simulate,
        title=f"Snippets vs RTMPC nominal ({tag}){' [re-simulated]' if simulate else ''}",
    )

    fig_pp, _ = plot_snippet_phase(
        rtmpc_ref, subset,
        simulate=simulate,
        title=f"Phase portrait ({tag}){' [re-simulated]' if simulate else ''}",
    )

    if out_dir is None:
        out_dir = Path(save_dir) / "debug" / "plots" / "snippet_inspection"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_path = out_dir / f"timeseries_{tag}{sim_tag}.png"
    pp_path = out_dir / f"phase_{tag}{sim_tag}.png"
    fig_ts.savefig(ts_path, dpi=150, bbox_inches="tight")
    fig_pp.savefig(pp_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ts)
    plt.close(fig_pp)
    print(f"Saved: {ts_path}")
    print(f"Saved: {pp_path}")


# ---------------------------------------------------------------------------
# Plot learned reward network phase portrait
# ---------------------------------------------------------------------------

def plot_learned_reward_network(
    save_dir: str | Path = DEFAULT_SAVE_DIR,
    device: str = "cuda",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the learned reward network phase portrait."""

    # Set up double integrator environment
    env_id = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0"
    venv = gym.make(env_id)

    # Load the learned reward network
    reward_net_path = Path(save_dir) / "reward_net" / "reward_net_state.pth"
    if not reward_net_path.exists():
        raise FileNotFoundError(f"Reward network not found at {reward_net_path}")

    reward_net = TrajectoryRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_state=True,
        use_action=False,
        use_next_state=False,
        use_done=False,
        hid_sizes=(256, 256),
    )
    reward_net.load_state_dict(th.load(reward_net_path, map_location=device))
    reward_net.to(device)

    # Build a meshgrid over (position, velocity) — the full 2D state space.
    n_grid = 100
    pos_vals = np.linspace(venv.observation_space.low[0], venv.observation_space.high[0], n_grid)
    vel_vals = np.linspace(venv.observation_space.low[1], venv.observation_space.high[1], n_grid)
    pos_grid, vel_grid = np.meshgrid(pos_vals, vel_vals)  # each (n_grid, n_grid)

    # Flatten into (n_grid^2, 2) state matrix and evaluate R_θ(s).
    states_flat = np.stack([pos_grid.ravel(), vel_grid.ravel()], axis=1).astype(np.float32)
    with th.no_grad():
        reward_flat = reward_net(th.from_numpy(states_flat).to(device))
    reward_grid = reward_flat.cpu().numpy().reshape(n_grid, n_grid)

    # Plot as a filled contour over (position, velocity).
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(pos_grid, vel_grid, reward_grid, levels=50, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="R_θ(s)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("Learned reward contour R_θ(position, velocity)")

    out_path = Path(save_dir) / "reward_net_contour.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    return fig, ax
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Double integrator debug helpers.",
    )
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("inspect-snippets", help="Plot augmented snippets vs RTMPC nominal")
    sp.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR))
    sp.add_argument("--noise-level", type=float, default=0.0)
    sp.add_argument("--traj-idx", type=int, default=0)
    sp.add_argument("--n-snippets", type=int, default=10)
    sp.add_argument("--simulate", action="store_true")
    sp.add_argument("--state-idx", type=int, default=0)
    sp.add_argument("--out-dir", type=str, default=None)

    sp = sub.add_parser("plot-reward-network", help="Plot the learned reward network phase portrait")
    sp.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR))
    sp.add_argument("--device", type=str, default="cuda")

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.command == "inspect-snippets":
        inspect_snippets(
            save_dir=args.save_dir,
            noise_level=args.noise_level,
            traj_idx=args.traj_idx,
            n_snippets=args.n_snippets,
            simulate=args.simulate,
            state_idx=args.state_idx,
            out_dir=args.out_dir,
        )
    elif args.command == "plot-reward-network":
        plot_learned_reward_network(
            save_dir=args.save_dir,
            device=args.device,
        )
    else:
        plot_learned_reward_network(
            save_dir=DEFAULT_SAVE_DIR,
            device="cpu",
        )
        # _build_parser().print_help()

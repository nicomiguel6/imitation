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
DREX_SAVE_DIR = SCRIPT_DIR / "drex_outputs"


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
    reference_trajectory: Optional[types.Trajectory] = None,
    noise_level: float = 0.0,
    traj_idx: int = 0,
    ref_pos: float = 0.0,
    ref_vel: float = 0.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the learned reward contour with an optional reference trajectory overlay.

    The observation fed to the network is [pos, vel, ref_pos, ref_vel].  The
    meshgrid sweeps (pos, vel) while ref_pos and ref_vel are held constant,
    which is appropriate when testing with a fixed target.

    Args:
        save_dir: NTRILTrainer output directory.
        device: Torch device string.
        reference_trajectory: Trajectory whose states will be overlaid on the
            contour.  If None, the RTMPC nominal trajectory for *noise_level*
            and *traj_idx* is loaded automatically from *save_dir*.
        noise_level: Noise bin used to load the reference trajectory when
            *reference_trajectory* is None.
        traj_idx: Index into the RTMPC trajectory list for that noise bin.
        ref_pos: Constant reference position appended to every grid state.
        ref_vel: Constant reference velocity appended to every grid state.
    """
    env_id = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0"
    venv = gym.make(env_id)

    # Load the learned reward network ensemble.
    ensemble = []
    n_ensemble = 3
    for i in range(n_ensemble):
        reward_net_path = Path(save_dir) / "ensemble" / f"reward_net_{i}.pth"
        net = TrajectoryRewardNet(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            hid_sizes=(256, 256),
        )
        net.load_state_dict(th.load(str(reward_net_path), map_location=device))
        net.to(device)
        net.eval()
        ensemble.append(net)

    # Build a meshgrid over (position, velocity) — the first two state dims.
    # ref_pos and ref_vel are held constant to match the evaluation condition.
    n_grid = 100
    pos_vals = np.linspace(venv.observation_space.low[0], venv.observation_space.high[0], n_grid)
    vel_vals = np.linspace(venv.observation_space.low[1], venv.observation_space.high[1], n_grid)
    pos_grid, vel_grid = np.meshgrid(pos_vals, vel_vals)

    n_pts = n_grid * n_grid
    states_flat = np.stack(
        [
            pos_grid.ravel(),
            vel_grid.ravel(),
            np.full(n_pts, ref_pos),
            np.full(n_pts, ref_vel),
        ],
        axis=1,
    ).astype(np.float32)
    action_dim = venv.action_space.shape[0]
    dummy_action = np.zeros((n_pts, action_dim), dtype=np.float32)
    dummy_next = np.zeros_like(states_flat)
    dummy_done = np.zeros(n_pts, dtype=np.float32)

    per_net = np.stack(
        [net.predict_processed(states_flat, dummy_action, dummy_next, dummy_done) for net in ensemble],
        axis=0,
    )
    reward_grid = per_net.mean(axis=0).reshape(n_grid, n_grid)

    # Resolve the reference trajectory.
    if reference_trajectory is None:
        rtmpc_trajs, _ = load_augmented_data(save_dir, noise_level)
        if traj_idx >= len(rtmpc_trajs):
            raise IndexError(
                f"traj_idx={traj_idx} but only {len(rtmpc_trajs)} RTMPC trajectories available"
            )
        reference_trajectory = rtmpc_trajs[traj_idx]

    # Plot contour.
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(pos_grid, vel_grid, reward_grid, levels=50, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="R_θ(s)")

    # # Overlay the reference trajectory.
    # traj_pos = reference_trajectory.obs[:, 0]
    # traj_vel = reference_trajectory.obs[:, 1]
    # ax.plot(traj_pos, traj_vel, "w-", linewidth=1.5, label="Reference trajectory")
    # ax.plot(traj_pos[0], traj_vel[0], "wo", markersize=7, label="Start")
    # ax.plot(traj_pos[-1], traj_vel[-1], "w*", markersize=10, label="End")

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Learned reward contour  |  noise={noise_level:.2f}, traj={traj_idx}")
    ax.legend(loc="upper right", fontsize=8)

    out_path = Path(save_dir) / "reward_net_contour.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    return fig, ax


# ---------------------------------------------------------------------------
# Investigate ranked dataset
# ---------------------------------------------------------------------------

def measure_ranking_separability(
    ensemble_dir: str | Path = DEFAULT_SAVE_DIR,
) -> None:
    """Measure the ranking separability of the ranked dataset. Check each labeled snippet pair's reward difference."""

    # Load all ranked datasets paths 
    n_ensemble = 3
    ranked_dataset_paths = [Path(ensemble_dir) / "ensemble" / f"ranked_samples_{i}.pth" for i in range(n_ensemble)]

    
    # define reward function using MPC cost
    Q = np.diag([10.0, 1.0])
    def reward_function(states):
        reward_tmp = []
        for state in states:
            state = state.flatten()
            state = state[:2]
            reward_tmp.append(-state.T @ Q @ state)
        return np.sum(reward_tmp).flatten()[0] # should be a scalar
    

    # Set up reward storing dictionary
    reward_stats = {i: {'accuracy': [], 'reward': [], 'correct_count': 0, 'incorrect_count': 0} for i in range(n_ensemble)}
    # Check each labeled snippet pair's reward difference
    for idx, ranked_dataset_path in enumerate(ranked_dataset_paths):
        ranked_dataset = th.load(str(ranked_dataset_path))
        for snippet_pair, label in ranked_dataset['samples']:
            reward_diff = reward_function(snippet_pair[0]) - reward_function(snippet_pair[1])
            reward_stats[idx]['reward'].append(reward_diff)
            if reward_diff > 0 and label == 0: # if reward_diff is positive and label is 0, then the first snippet is better and this is correct
                reward_stats[idx]['correct_count'] += 1
            elif reward_diff < 0 and label == 1: # if reward_diff is negative and label is 1, then the second snippet is better and this is correct
                reward_stats[idx]['correct_count'] += 1
            elif reward_diff > 0 and label == 1: # if reward_diff is positive and label is 1, then the first snippet is better and this is incorrect
                reward_stats[idx]['incorrect_count'] += 1
            elif reward_diff < 0 and label == 0: # if reward_diff is negative and label is 0, then the second snippet is better and this is incorrect
                reward_stats[idx]['incorrect_count'] += 1
        reward_stats[idx]['accuracy'].append(reward_stats[idx]['correct_count'] / (reward_stats[idx]['correct_count'] + reward_stats[idx]['incorrect_count']))

    # Print the reward stats
    for i in range(n_ensemble):
        print(f"Ensemble {i}: Accuracy: {reward_stats[i]['accuracy']}, Correct count: {reward_stats[i]['correct_count']}, Incorrect count: {reward_stats[i]['incorrect_count']}")

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
    sp.add_argument("--noise-level", type=float, default=0.0,
                    help="Noise bin for the reference trajectory overlay")
    sp.add_argument("--traj-idx", type=int, default=0,
                    help="Which RTMPC trajectory to overlay")
    sp.add_argument("--ref-pos", type=float, default=0.0,
                    help="Constant reference position appended to grid states (default: 0.0)")
    sp.add_argument("--ref-vel", type=float, default=0.0,
                    help="Constant reference velocity appended to grid states (default: 0.0)")

    sp = sub.add_parser("measure-ranking-separability", help="Measure the ranking separability of the ranked dataset")
    sp.add_argument("--ensemble-dir", type=str, default=str(DEFAULT_SAVE_DIR))

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
            noise_level=args.noise_level,
            traj_idx=args.traj_idx,
            ref_pos=args.ref_pos,
            ref_vel=args.ref_vel,
        )
    elif args.command == "measure-ranking-separability":
        measure_ranking_separability(
            ensemble_dir=args.ensemble_dir,
        )
    else:
        # DREX_SAVE_DIR = SCRIPT_DIR / "drex_outputs"
        # NTRIL_SAVE_DIR = SCRIPT_DIR / "ntril_outputs"
        reference_trajectory = np.load(DEFAULT_SAVE_DIR / "reference_trajectory.npy")
        plot_learned_reward_network(
            save_dir=DEFAULT_SAVE_DIR,
            device="cpu",
            reference_trajectory=reference_trajectory,
            ref_pos=2.0,
            ref_vel=0.0,
        )
        # measure_ranking_separability()
        # _build_parser().print_help()

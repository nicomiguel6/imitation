"""Compare the suboptimal PID expert against MPC on the double integrator."""

from pathlib import Path

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC
from stable_baselines3 import PPO, SAC
from imitation.scripts.NTRIL.double_integrator.double_integrator import DoubleIntegratorSuboptimalPolicy, generate_reference_trajectory
from imitation.data import types


SCRIPT_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Model selection — add or remove names from this list to control which
# models are simulated.  Available keys (defined in build_models() below):
#   "ntril"       – NTRIL final policy (PPO)
#   "drex"        – Pure D-REX final policy (PPO)
#   "suboptimal"  – Suboptimal PID expert
#   "mpc"         – Robust Tube MPC
# ---------------------------------------------------------------------------
ACTIVE_MODELS = [
    # "ntril",
    # "drex",
    "suboptimal",
    # "mpc",
    "airl",
#     "ssrr",
]

# ---------------------------------------------------------------------------
# Per-model display settings used when plotting.
# ---------------------------------------------------------------------------
MODEL_STYLE = {
    "ntril":      {"color": "k", "linestyle": "-",  "label": "NTRIL Final Policy"},
    "drex":       {"color": "g", "linestyle": "-",  "label": "DREX Policy"},
    "suboptimal": {"color": "b", "linestyle": "--", "label": "Suboptimal Policy"},
    "mpc":        {"color": "r", "linestyle": "-",  "label": "MPC"},
    "airl":       {"color": "y", "linestyle": "--", "label": "AIRL"},
    "ssrr":       {"color": "c", "linestyle": "-",  "label": "SSRR"},
}


def build_models(active_models, dt, max_episode_seconds, disturbance_magnitude, disturbance_vertices, reference_trajectory):
    """Instantiate only the requested models and their environments.

    Returns a dict keyed by model name, each value being a dict with:
        env     – the Gymnasium environment
        predict – callable(obs) -> action
        reset   – optional callable(obs) called once before each episode
    """
    env_kwargs = dict(
        max_episode_seconds=max_episode_seconds,
        dt=dt,
        disturbance_magnitude=disturbance_magnitude,
        reference_trajectory=reference_trajectory,
    )
    env_id = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0"

    models = {}

    if "ntril" in active_models:
        ntril_path = (
            SCRIPT_DIR / "ntril_outputs" / "final_policy" / "final_policy.zip"
        )
        policy = PPO.load(ntril_path, device="cuda")
        models["ntril"] = {
            "env": gym.make(env_id, **env_kwargs),
            "predict": lambda obs, p=policy: p.predict(obs)[0],
            "reset": None,
        }

    if "drex" in active_models:
        drex_path = SCRIPT_DIR / "drex_outputs" / "final_policy" / "final_policy.zip"
        policy = PPO.load(drex_path, device="cuda")
        models["drex"] = {
            "env": gym.make(env_id, **env_kwargs),
            "predict": lambda obs, p=policy: p.predict(obs)[0],
            "reset": None,
        }

    if "suboptimal" in active_models:
        env = gym.make(env_id, **env_kwargs)
        policy = DoubleIntegratorSuboptimalPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        policy.set_K_values(0.02, 0.3)
        models["suboptimal"] = {
            "env": env,
            "predict": policy._choose_action,
            "reset": None,
        }

    if "mpc" in active_models:
        env = gym.make(env_id, **env_kwargs)
        max_episode_steps = int(max_episode_seconds / dt)
        ref_traj_mpc = types.Trajectory(
            obs=reference_trajectory,
            acts=np.zeros((max_episode_steps, 1)),
            infos=np.array([{}] * max_episode_steps),
            terminal=True,
        )
        policy = RobustTubeMPC(
            horizon=10,
            time_step=dt,
            A=env.unwrapped.A_d,
            B=env.unwrapped.B_d,
            Q=np.diag([10.0, 1.0]),
            R=0.1 * np.eye(1),
            state_bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
            control_bounds=(np.array([-20.0]), np.array([20.0])),
            artificial_disturbance_vertices=np.array([[0.0], [0.0]]),
            disturbance_vertices=disturbance_vertices,
            reference_trajectory=ref_traj_mpc,
            use_approx=True,
        )
        policy.setup()
        policy.set_reference_trajectory(ref_traj_mpc)
        models["mpc"] = {
            "env": env,
            "predict": lambda obs, p=policy: p.solve_mpc(obs)[1],
            "reset": lambda obs, p=policy: p.reset_episode(obs[:p.state_dim]),
        }

    if "airl" in active_models:
        airl_path = "/home/nicomiguel/imitation/src/imitation/scripts/SSRR/tests/airl_outputs/20260501_224503_constant_P0.0/initial_BC_policy/bc_policy.zip" # for debugging initial BC policy
        # airl_path = "/home/nicomiguel/imitation/src/imitation/scripts/SSRR/tests/airl_outputs/20260501_221911_constant_P0.0/best_checkpoint/learner_policy.zip"
        # airl_path = "/home/nicomiguel/imitation/src/imitation/scripts/SSRR/tests/airl_outputs/final_policy/learner_policy.zip"
        # airl_path = "/home/nicomiguel/imitation/src/imitation/scripts/SSRR/tests/airl_outputs/20260420_224617_sinusoidal_A1.0_f0.01/best_checkpoint/learner_policy.zip"
        policy = PPO.load(airl_path, device="cpu")
        models["airl"] = {
            "env": gym.make(env_id, **env_kwargs),
            "predict": lambda obs, p=policy: p.predict(obs)[0],
            "reset": None,
        }

    if "ssrr" in active_models:
        ssrr_path = "/home/nicomiguel/imitation/src/imitation/scripts/SSRR/tests/airl_outputs/20260420_231943_sinusoidal_A1.0_f0.01/ssrr_rl/latest/ssrr_rl_policy.zip"
        policy = SAC.load(ssrr_path, device="cuda")
        models["ssrr"] = {
            "env": gym.make(env_id, **env_kwargs),
            "predict": lambda obs, p=policy: p.predict(obs)[0],
            "reset": None,
        }
    return models


def main():
    # Simulation params
    dt = 1.0
    max_episode_seconds = 200.0
    disturbance_magnitude = 0.0
    disturbance_vertices = np.array([
        [ disturbance_magnitude,  disturbance_magnitude],
        [-disturbance_magnitude, -disturbance_magnitude],
        [-disturbance_magnitude,  disturbance_magnitude],
        [ disturbance_magnitude, -disturbance_magnitude],
    ])
    max_episode_steps = int(max_episode_seconds / dt)

    # reference_trajectory = generate_reference_trajectory(
    #     T=max_episode_steps, dt=dt, mode="sinusoidal", amplitude=1.0, frequency=0.01, phase=0.0
    # )
    reference_trajectory = generate_reference_trajectory(
        T=max_episode_steps, dt=dt, mode="constant", target_position=0.0
    )

    models = build_models(
        ACTIVE_MODELS, dt, max_episode_seconds, disturbance_magnitude, disturbance_vertices, reference_trajectory
    )

    initial_state = np.random.uniform(-2.0, 2.0, size=(2,)).astype(np.float32)

    for j in range(1):
        # Reset all environments to the same initial state
        obs_per_model = {}
        for name, m in models.items():
            obs, _ = m["env"].reset(
                state=initial_state,
                options={"reference_trajectory": reference_trajectory},
            )
            obs_per_model[name] = obs
            if m["reset"] is not None:
                m["reset"](obs)

        states   = {name: [obs_per_model[name].copy()] for name in models}
        actions  = {name: [] for name in models}
        rewards  = {name: [] for name in models}
        dones    = {name: False for name in models}

        # Determine episode length from the first active env
        ref_env = next(iter(models.values()))["env"]
        for i in range(ref_env.unwrapped.max_episode_steps):
            for name, m in models.items():
                if dones[name]:
                    continue
                action = m["predict"](obs_per_model[name])
                obs, reward, terminated, truncated, _ = m["env"].step(action)
                obs_per_model[name] = obs
                states[name].append(obs)
                actions[name].append(action)
                rewards[name].append(reward)
                if terminated or truncated:
                    dones[name] = True

            if all(dones.values()):
                break

        # Plot
        fig, ax = plt.subplots()
        for name in models:
            traj = np.array(states[name]).reshape(-1, 4)
            style = MODEL_STYLE.get(name, {})
            ax.plot(
                traj[:, 0],
                color=style.get("color", None),
                linestyle=style.get("linestyle", "-"),
                label=style.get("label", name),
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.set_title("Policy Comparison in Noisy Environment")
        ax.legend()
        fig.savefig(SCRIPT_DIR / f"trajectory_comparison_noisy_{j}.png")
        plt.close()

        for name in models:
            print(f"Total {MODEL_STYLE.get(name, {}).get('label', name)} cost: {sum(rewards[name]):.4f}")


if __name__ == "__main__":
    main()

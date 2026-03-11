"""Compare the suboptimal PID expert against MPC on the double integrator."""

from pathlib import Path

import numpy as np
import torch as th
import gymnasium as gym
import matplotlib.pyplot as plt

from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC
from stable_baselines3 import PPO
from imitation.scripts.NTRIL.double_integrator.double_integrator import DoubleIntegratorSuboptimalPolicy
from imitation.data import types


SCRIPT_DIR = Path(__file__).parent.resolve()


def main():
    # Simulation params
    disturbance_magnitude = 0.1
    disturbance_vertices = np.array([[disturbance_magnitude, disturbance_magnitude], [-disturbance_magnitude, -disturbance_magnitude], [-disturbance_magnitude, disturbance_magnitude], [disturbance_magnitude, -disturbance_magnitude]])
    dt = 1.0
    max_episode_seconds = 200

    max_episode_steps = int(max_episode_seconds / dt)
    # Set up reference trajectory
    reference_trajectory = np.load(SCRIPT_DIR / "ntril_outputs" / "reference_trajectory.npy")
    reference_trajectory_mpc = types.Trajectory(obs=reference_trajectory, acts=np.zeros((max_episode_steps, 1)), infos=np.array([{}] * max_episode_steps), terminal=True)
    
    # Set up envs
    env_mpc = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0", max_episode_seconds=max_episode_seconds, dt = dt, disturbance_magnitude=0.1, reference_trajectory=reference_trajectory)
    env_policy = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0", max_episode_seconds=max_episode_seconds, dt = dt, disturbance_magnitude=0.1, reference_trajectory=reference_trajectory)
    env_suboptimal = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0", max_episode_seconds=max_episode_seconds, dt = dt, disturbance_magnitude=0.1, reference_trajectory = reference_trajectory)

    # # load final policy
    # final_policy_path = SCRIPT_DIR / "ntril_outputs" / "final_policy" / "final_policy.zip"
    # final_policy = PPO.load(final_policy_path, device="cuda")

    # load pure drex policy
    drex_policy_path = SCRIPT_DIR / "drex_outputs" / "final_policy" / "final_policy.zip"
    final_policy = PPO.load(drex_policy_path, device="cuda")

    # Suboptimal policy
    suboptimal_policy = DoubleIntegratorSuboptimalPolicy(
        observation_space=env_policy.observation_space,
        action_space=env_policy.action_space,
    )

    device = "cuda"
    if device == "mps":
        th.set_default_dtype(th.float32)
    else:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

    mpc_policy = RobustTubeMPC(
        horizon=10,
        time_step=dt,
        A=env_mpc.A_d,
        B=env_mpc.B_d,
        Q=np.diag([10.0, 1.0]),
        R=0.1 * np.eye(1),
        state_bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
        control_bounds=(np.array([-2.0]), np.array([2.0])),
        disturbance_bound = disturbance_magnitude,
        disturbance_vertices = disturbance_vertices,
        reference_trajectory = reference_trajectory_mpc,
    )
    mpc_policy.setup()

    mpc_policy.set_reference_trajectory(reference_trajectory_mpc)
    K = [0.02, 0.3]
    suboptimal_policy.set_K_values(K[0], K[1])

    initial_state = np.random.uniform(-5.0, 5.0, size=(2,)).astype(np.float32)

    for j in range(1):
        obs, info = env_policy.reset(state=initial_state, options={"reference_trajectory": reference_trajectory})
        obs_mpc, info = env_mpc.reset(state=initial_state, options={"reference_trajectory": reference_trajectory})
        obs_suboptimal, info = env_suboptimal.reset(state=initial_state, options={"reference_trajectory": reference_trajectory})
        states_policy = [obs.copy()]
        states_mpc = [obs.copy()]
        states_suboptimal = [obs_suboptimal.copy()]
        actions_policy = []
        actions_mpc = []
        actions_suboptimal = []
        rewards_policy = []
        rewards_mpc = []
        rewards_suboptimal = []
        mpc_policy.reset_episode(obs[:mpc_policy.state_dim])
        for i in range(env_policy.unwrapped.max_episode_steps):
            action_suboptimal = suboptimal_policy._choose_action(obs_suboptimal)
            action_policy, _ = final_policy.predict(obs)
            _, action_mpc, _ = mpc_policy.solve_mpc(obs_mpc)
            obs, reward, terminated, truncated, info = env_policy.step(action_policy)
            obs_mpc, reward_mpc, terminated_mpc, truncated_mpc, info_mpc = env_mpc.step(action_mpc)
            obs_suboptimal, reward_suboptimal, terminated_suboptimal, truncated_suboptimal, info_suboptimal = env_suboptimal.step(action_suboptimal)
            states_policy.append(obs)
            states_mpc.append(obs_mpc)
            states_suboptimal.append(obs_suboptimal)
            actions_policy.append(action_policy)
            actions_mpc.append(action_mpc)
            actions_suboptimal.append(action_suboptimal)
            rewards_policy.append(reward)
            rewards_mpc.append(reward_mpc)
            rewards_suboptimal.append(reward_suboptimal)
            if terminated or truncated:
                break


        states_policy = np.array(states_policy).reshape(-1, 4)
        states_mpc = np.array(states_mpc).reshape(-1, 4)
        states_suboptimal = np.array(states_suboptimal).reshape(-1, 4)
        fig, ax = plt.subplots()
        ax.plot(states_policy[:, 0], "k-", label="Final Policy Trajectory")
        ax.plot(states_mpc[:, 0], "r-", label="MPC Trajectory")
        ax.plot(states_suboptimal[:, 0], "b-", label="Suboptimal Policy Trajectory")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.set_title("Final Policy, MPC, and Suboptimal Policy in Noisy Environment")
        ax.legend()
        fig.savefig(SCRIPT_DIR / f"trajectory_comparison_noisy_{j}.png")
        plt.close()

        print("Total final policy cost: ", sum(rewards_policy))
        print("Total MPC cost: ", sum(rewards_mpc))
        print("Total suboptimal policy cost: ", sum(rewards_suboptimal))

if __name__ == "__main__":
    main()

import numpy as np
import gymnasium as gym
from imitation.scripts.NTRIL.double_integrator.double_integrator import (
    DoubleIntegratorEnv,
)
import matplotlib.pyplot as plt

env = gym.make("DoubleIntegrator-v0", target_position=1.0)
obs, info = env.reset()

# Store initial state
initial_state = obs

# Store states and actions for plotting
states = [obs.copy()]
actions = []
rewards = []

for i in range(env.unwrapped.max_episode_steps):
    action = env.unwrapped.suboptimal_expert(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    states.append(obs)
    actions.append(action)
    rewards.append(reward)
    if terminated or truncated:
        break

# print final observation
print("Final state: ", obs)
print("Final reward: ", sum(rewards))

# Plot position over time
states = np.array(states)
actions = np.array(actions)

time_steps = np.arange(states.shape[0])
plt.plot(initial_state[0], initial_state[1], "b+", label="initial position")
plt.plot(states[:, 0], states[:, 1], "k-", label="trajectory")
plt.xlabel("position")
plt.ylabel("velocity")
plt.legend()
plt.show()

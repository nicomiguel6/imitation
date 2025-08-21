# Noisy Trajectory Ranked Imitation Learning (NTRIL)

This folder contains scripts and resources for the Noisy Trajectory Ranked Imitation Learning (NTRIL) pipeline. The purpose is to systematically generate and process demonstration data with varying noise levels, enabling robust imitation learning and reward inference.

## Pipeline Overview

1. **Behavioral Cloning (BC):**  
    Train a policy using standard behavioral cloning on existing demonstration data.

2. **Noisy Rollout Generation:**  
    Apply increasing levels of noise to the BC policy and collect trajectory rollouts for each noise level.

3. **Robust Tube MPC & Data Augmentation:**  
    For each noisy rollout, solve a linear robust tube Model Predictive Control (MPC) problem under a specified disturbance set. Augment the collected state-action pairs using Tagliabue's method.

4. **Ranked Dataset Construction:**  
    Build a comprehensive dataset where samples are ranked according to the level of noise present in their generation.

5. **Demonstration Ranked IRL:**  
    Use the ranked dataset to perform Inverse Reinforcement Learning (IRL), learning a reward network that reflects the ranking structure.

6. **Policy Fitting via RL:**  
    Fit an appropriate policy to the learned reward network using Reinforcement Learning (RL) techniques.

---
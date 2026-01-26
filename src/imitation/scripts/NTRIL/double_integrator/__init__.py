from gymnasium.envs import register


# Register the environment with gymnasium
register(
    id="DoubleIntegrator-v0",
    entry_point="imitation.scripts.NTRIL.double_integrator.double_integrator:DoubleIntegratorEnv",
    max_episode_steps=200,
)

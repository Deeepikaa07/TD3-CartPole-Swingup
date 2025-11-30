import gymnasium as gym
import shimmy
import numpy as np
from dm_control import suite
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise


seeds = [0, 1, 2]

for seed in seeds:
    env = gym.make("dm_control/cartpole-swingup-v0")
    env = Monitor(env, filename=f"logs/seed_{seed}")

    # TD3 needs action noise for exploration (deterministic policy)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        "MultiInputPolicy",      # same as SAC since dm_control obs are dict-like
        env,
        action_noise=action_noise,
        seed=seed,
        device="cuda:0",
        verbose=1,
    )
    model.learn(total_timesteps=200000, log_interval=4)

    model.save(f"weights/td3_cartpole_seed{seed}")

    env.close()

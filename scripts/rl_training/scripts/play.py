import rl_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    env = rl_env.StrategySimEnv(action_hz=10, enable_rendering=True)
    model = PPO.load("ppo_strategy_sim", env=env)
    env.enable_rendering()
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, dones, info = env.step(action)
    env.close()

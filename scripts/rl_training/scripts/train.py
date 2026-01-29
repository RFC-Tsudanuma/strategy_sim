import rl_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env():
    return rl_env.StrategySimEnv(action_hz=10)


if __name__ == "__main__":
    vec_env = make_vec_env(make_env, n_envs=64, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")
    print("start learning...")
    import time

    start = time.perf_counter()
    model.learn(total_timesteps=2000000)
    elapsed = time.perf_counter() - start
    print(f"learn elapsed time: {elapsed:.2f}")
    model.save("ppo_strategy_sim")
    vec_env.close()

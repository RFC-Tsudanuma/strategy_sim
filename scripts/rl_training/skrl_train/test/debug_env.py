import os
import sys

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../scripts")
)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../scripts"))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../.."))

from env_config import StrategyEnvConfig
from numpy import set_printoptions

if __name__ == "__main__":
    env = StrategyEnvConfig(
        action_hz=10, max_episode_length=3000, seed=42, complete_observation=True
    )
    obs = env.reset()
    env.render("human")
    set_printoptions(precision=4, suppress=True)
    print(obs)
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, done, info = env.step(action)
        print(
            f"obs: {obs}, reward: {reward}, terminated: {terminated}, done: {done}, info: {info}"
        )
        if terminated or done:
            obs = env.reset()
            print("reset")

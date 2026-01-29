import os
import sys

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../scripts")
)
import argparse

import yaml

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--num_envs", type=int, default=16, help="number of parallel environments"
)
argparser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")
argparser.add_argument("--play", action="store_true", help="play mode")

args = argparser.parse_args()

import conv_tensor
import models as rl_models
from env_config import StrategyEnvConfig
from gymnasium.vector import AsyncVectorEnv
from rl_env import StrategySimEnv
from skrl.agents.torch.ppo import PPO_RNN as PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import gaussian_model
from torch import inference_mode, tensor

yaml_cfg = yaml.safe_load(
    open(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "skrl_ppo_lstm_cfg.yaml"
        ),
        "r",
    )
)


def make_env():
    return StrategyEnvConfig(action_hz=10, **yaml_cfg["env"])


def main():
    if args.play:
        vec_env = StrategySimEnv(
            action_hz=10, complete_observation=yaml_cfg["env"]["complete_observation"]
        )
    else:
        vec_env = AsyncVectorEnv([lambda: make_env() for _ in range(args.num_envs)])

    # wrap the environment
    env = wrap_env(vec_env, wrapper="gymnasium")

    print(f"Target device = {env.device}")
    print(f"num_envs = {env.num_envs}")

    # instantiate the memory (assumes there is a wrapped environment: env)
    memory = RandomMemory(
        memory_size=yaml_cfg["memory"]["memory_size"],
        num_envs=env.num_envs,
        device=env.device,
    )

    models = {}
    models["policy"] = rl_models.LSTM(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        num_envs=env.num_envs,
        **yaml_cfg["models"]["policy"],
    )
    models["value"] = gaussian_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        **yaml_cfg["models"]["value"],
    )

    yaml_cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveRL
    agent = PPO(
        models=models,  # models dict
        memory=memory,  # memory instance, or None if not required
        cfg=yaml_cfg[
            "agent"
        ],  # configuration dict (preprocessors, learning rate schedulers, etc.)
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    if args.play:
        if args.checkpoint is None:
            print("Please provide checkpoint path")
            quit()
        agent.init()
        agent.load(args.checkpoint)
        agent.set_running_mode("eval")
        from time import perf_counter

        obs, _ = vec_env.reset()
        vec_env.enable_rendering()
        timestep = 0
        inf_time = 0.0
        while True:
            start = perf_counter()
            with inference_mode():
                torch_obs = tensor(obs, device=env.device)
                outputs = agent.act(torch_obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                actions = conv_tensor.convert_tensor_to_np(actions)
            inf_time += perf_counter() - start
            obs, _, _, _, _ = vec_env.step(actions)
            timestep += 1
            if timestep % 100 == 0:
                print(f"Step {timestep}, inference time: {inf_time / 100:.8f}")
                inf_time = 0.0
    else:
        trainer = SequentialTrainer(env=env, agents=[agent], cfg=yaml_cfg["trainer"])
        trainer.train()


if __name__ == "__main__":
    main()

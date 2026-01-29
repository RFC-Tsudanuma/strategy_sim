import os
import sys

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../scripts")
)

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--num_envs", type=int, default=16, help="number of parallel environments"
)
argparser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")
args_cli, hydra_args = argparser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.checkpoint is None:
    print("Please provide checkpoint path")
    quit()

import conv_tensor
import env_config
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.runner.torch import Runner


@hydra.main(
    config_name="skrl_ppo_cfg",
    version_base=None,
    config_path=os.path.abspath(os.path.dirname(__file__)),
)
def main(cfg: DictConfig):
    # load the experiment config and instantiate the runner
    cfg = OmegaConf.to_container(
        cfg, resolve=True
    )  # Runnerのために普通のdictに変換する

    sim_env = env_config.StrategyEnvConfig(
        action_hz=10,
        complete_observation=cfg["complete_observation"],
    )

    print(f"Running with {args_cli.num_envs} environments")
    cfg["agent"]["experiment"]["directory"] = "play_log"
    env = wrap_env(sim_env, wrapper="gymnasium")
    runner = Runner(env, cfg)

    # # load a checkpoint to continue training or for evaluation (optional)
    runner.agent.load(args_cli.checkpoint)

    # run the training
    runner.agent.set_running_mode("eval")

    from time import perf_counter

    obs, _ = sim_env.reset()
    sim_env.enable_rendering()
    timestep = 0
    inf_time = 0.0
    while True:
        start = perf_counter()
        with torch.inference_mode():
            torch_obs = torch.tensor(obs, device=env.device)
            outputs = runner.agent.act(torch_obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            actions = conv_tensor.convert_tensor_to_np(actions)
        inf_time += perf_counter() - start
        obs, _, _, _, _ = sim_env.step(actions)
        timestep += 1
        if timestep % 100 == 0:
            print(f"Step {timestep}, inference time: {inf_time / 100:.8f}")
            inf_time = 0.0


if __name__ == "__main__":
    main()

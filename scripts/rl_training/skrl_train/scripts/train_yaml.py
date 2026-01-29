import os
import sys

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../scripts")
)
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--num_envs", type=int, default=None, help="number of parallel environments"
)
argparser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")
default_conf_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "skrl_ppo_cfg.yaml"
)
argparser.add_argument(
    "--config_path", type=str, default=default_conf_path, help="config yaml path"
)
argparser.add_argument("--use_async", action="store_true", help="using AsyncVectorEnv")

args_cli, hydra_args = argparser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

import env_config
import hydra
from gymnasium.vector import SyncVectorEnv
from omegaconf import DictConfig, OmegaConf
from reward_log_wrapper import RewardLogAsyncVectorEnv, RewardLogSyncVectorEnv
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.runner.torch import Runner


@hydra.main(
    config_name="skrl_ppo_cfg",
    version_base=None,
    config_path=os.path.abspath(os.path.dirname(__file__)),
)
def main(cfg: DictConfig):
    if args_cli.num_envs is None:
        if os.cpu_count() is None:
            num_envs = 64
        else:
            num_envs = os.cpu_count() * 4
    else:
        num_envs = args_cli.num_envs
    cfg = OmegaConf.to_container(
        cfg, resolve=True
    )  # Runnerのために普通のdictに変換する
    if cfg["seed"] is None:
        seed = None
    else:
        seed = int(cfg["seed"])

    def make_env():
        return env_config.StrategyEnvConfig(
            reward_cfg=cfg["rewards"],
            action_hz=10,
            seed=seed,
            max_episode_length=cfg["episode_length"],
            complete_observation=cfg["complete_observation"],
        )

    if args_cli.use_async:
        vec_env = RewardLogAsyncVectorEnv([lambda: make_env() for _ in range(num_envs)])
    else:
        vec_env = RewardLogSyncVectorEnv([lambda: make_env() for _ in range(num_envs)])

    env = wrap_env(vec_env, wrapper="gymnasium")

    print(f"Running with {num_envs} environments")
    print(f"Running on device {env.device}")

    # load the experiment config and instantiate the runner
    runner = Runner(env, cfg)

    if args_cli.checkpoint is not None:
        print(f"Loading checkpoint from {args_cli.checkpoint}")
        runner.agent.load(args_cli.checkpoint)

    # run the training
    runner.run("train")


if __name__ == "__main__":
    main()

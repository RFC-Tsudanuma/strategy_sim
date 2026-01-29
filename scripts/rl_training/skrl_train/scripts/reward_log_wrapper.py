from random import randint

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch import tensor


class RewardLogAsyncVectorEnv(AsyncVectorEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)

    def step(self, actions):
        observations, rewards, terminateds, truncateds, infos = super().step(actions)
        """
        infosの形式
        infos = [
            {
                "episode": {
                    "reward_scored_instant": array([tensor(0.0)]),
                    "reward_out_of_field_instant": array([tensor(0.0)]),
                    ... ここをarray -> tensorに変換する必要ある
                }
            },
            ...
        """
        acumed_infos = {"reward_info": {}}
        for k, v in infos.items():
            if k != "reward_info":
                continue
            for key, value in v.items():
                if key.startswith("reward_"):
                    idx = randint(0, self.num_envs - 1)
                    # 高速化のためにランダムにサンプリングした1つを標本値として使う
                    acumed_infos[k][key] = (
                        tensor(float(value[idx].item()))
                        if value[idx] is not None
                        else tensor(0.0)
                    )  # 返すのはtensorのインスタンスでないといけない
        return observations, rewards, terminateds, truncateds, acumed_infos


class RewardLogSyncVectorEnv(SyncVectorEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)

    def step(self, actions):
        observations, rewards, terminateds, truncateds, infos = super().step(actions)
        """
        infosの形式
        infos = [
            {
                "episode": {
                    "reward_scored_instant": array([tensor(0.0)]),
                    "reward_out_of_field_instant": array([tensor(0.0)]),
                    ... ここをarray -> tensorに変換する必要ある
                }
            },
            ...
        """
        acumed_infos = {"reward_info": {}}
        for k, v in infos.items():
            if k != "reward_info":
                continue
            for key, value in v.items():
                if key.startswith("reward_"):
                    idx = randint(0, self.num_envs - 1)
                    # 高速化のためにランダムにサンプリングした1つを標本値として使う
                    acumed_infos[k][key] = (
                        tensor(float(value[idx].item()))
                        if value[idx] is not None
                        else tensor(0.0)
                    )  # 返すのはtensorのインスタンスでないといけない
        return observations, rewards, terminateds, truncateds, acumed_infos

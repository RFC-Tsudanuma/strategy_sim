import os
import sys

from numpy import arctan2, cos, fabs, float32, random, sqrt
from torch import norm, tensor
from torch.nn.functional import cosine_similarity

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../scripts")
)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../.."))

import reward_utils
import setting
from randomize import ObservationRandomizerBase
from rl_env import StrategySimEnv
from training_interface import DomainRandomizer, GameInfo, RobotData, pick_object


def robot_out_of_field(observation: RobotData) -> bool:
    if observation.selfpos_x > setting.WIDTH / 2 + 50:
        return True
    if observation.selfpos_y > setting.HEIGHT / 2 + 50:
        return True
    if observation.selfpos_x < -setting.WIDTH / 2 - 50:
        return True
    if observation.selfpos_y < -setting.HEIGHT / 2 - 50:
        return True
    return False


class StrategyEnvConfig(StrategySimEnv):
    def __init__(
        self,
        reward_cfg: dict,
        action_hz=10,
        max_episode_length=3000,
        seed=None,
        complete_observation=False,
    ):
        super().__init__(
            action_hz=action_hz,
            max_episode_len=max_episode_length,
            seed=seed,
            randomizer=DRConfig(seed),
            obs_randomizer=TrueValueRandomizer(seed=seed),
            complete_observation=complete_observation,
        )
        self.scored = False
        self.reward_func_list: list = [
            attr for attr in dir(self) if attr.startswith("reward_")
        ]
        self.reward_cfg: dict = reward_cfg

    def calc_reward(
        self, observation: list[RobotData], game_state: GameInfo
    ) -> tuple[float, dict]:
        # 報酬関数を自動で全て呼ぶ
        reward = 0.0
        reward_details = {}
        # アクションを実行する時だけ記録する
        for func_name in self.reward_func_list:
            tmp = (
                getattr(self, func_name)(observation[self.target_robot - 1], game_state)
                * self.reward_cfg[func_name]["weight"]
            )
            reward_details[func_name] = tensor(tmp)
            reward += tmp
        self.prev_game_state = game_state
        return reward, reward_details

    def terminated(self, observation: list[RobotData], game_state: GameInfo) -> bool:
        if self.scored:
            # 敵味方に関わらず得点したら終了
            self.scored = False
            return True
        return robot_out_of_field(observation[self.target_robot - 1])

    def reward_scored(self, observation: RobotData, game_state: GameInfo) -> float:
        if self.prev_game_state is not None:
            # 得点が入った場合の報酬
            if game_state.our_score > self.prev_game_state.our_score:
                self.scored = True
                return 10000.0  # 味方が得点した場合
            if game_state.opponent_score > self.prev_game_state.opponent_score:
                self.scored = True
                return -10000.0  # 敵が得点した場合
        return 0.0

    def reward_out_of_field(
        self, observation: RobotData, game_state: GameInfo
    ) -> float:
        if robot_out_of_field(observation):
            return -100.0
        return 0.0

    def reward_ball_angle_to_robot(
        self, observation: RobotData, game_state: GameInfo
    ) -> float:
        ball = pick_object(observation.object_list, setting.ObjectLabel.LABEL_BALL)
        if ball != []:
            angle = arctan2(ball[0].object_y, ball[0].object_x)
            return cos(angle) * 1.0
        return 0.0

    def reward_last_action_vec_to_ball(
        self, observation: RobotData, game_state: GameInfo
    ) -> float:
        walk_cmd = self.last_action[0]
        ball = pick_object(observation.object_list, setting.ObjectLabel.LABEL_BALL)
        if ball != []:
            ball_vec = tensor([ball[0].object_x, ball[0].object_y])
            action_vec = tensor([walk_cmd.x_velocity, walk_cmd.y_velocity])
            cos_sim = cosine_similarity(action_vec.unsqueeze(0), ball_vec.unsqueeze(0))
            return cos_sim.item() * 1.0
        return 0.0

    # def reward_last_action_angle_to_ball(self, observation: RobotData, game_state: GameInfo) -> float:
    #     walk_cmd = self.last_action[0]
    #     ball = pick_object(observation.object_list, setting.ObjectLabel.LABEL_BALL)
    #     if ball != []:
    #         angle_to_ball = arctan2(ball[0].object_y, ball[0].object_x)
    #         cmd_theta = walk_cmd.theta_velocity
    #         # 等号が同じなら正の報酬、異なるなら0報酬
    #         return (cmd_theta * angle_to_ball >= 0) * 0.05
    #     return 0.0

    def reward_robot_vel_to_goal(
        self, observation: RobotData, game_state: GameInfo
    ) -> float:
        walk_cmd = self.last_action[0]
        goal_x = setting.GOALPOST_RIGHT_UP[0]
        goal_y = 0.0
        vec_to_goal = tensor(
            [goal_x - observation.selfpos_x, goal_y - observation.selfpos_y]
        )
        action_vec = reward_utils.robot_velocity_vec(
            walk_cmd.x_velocity, walk_cmd.y_velocity, observation.selfpos_theta
        )
        cos_sim = (
            cosine_similarity(action_vec.unsqueeze(0), vec_to_goal.unsqueeze(0)) * 2.0
        )
        return cos_sim.item() * 1.0

    def reward_rotation_penalty(
        self, observation: RobotData, game_state: GameInfo
    ) -> float:
        walk_cmd = self.last_action[0]
        # 大きく回転しようとする行動に対してペナルティを与える
        return -fabs(walk_cmd.theta_velocity) * 1.0


class DRConfig(DomainRandomizer):
    class BallRandomizer:
        def __init__(self, rng):
            self.rng = rng

        def randomize(self) -> tuple[float, float]:
            x = self.rng.uniform(0, setting.WIDTH)
            y = self.rng.uniform(0, setting.HEIGHT)
            return x, y

    class RobotRandomizer:
        def __init__(self, rng):
            self.rng = rng

        def randomize(self) -> tuple[float, float, float]:
            x = self.rng.uniform(0, setting.WIDTH / 2)
            y = self.rng.uniform(-setting.HEIGHT, 0)
            w = self.rng.uniform(-1.57, 1.57)
            return x, y, w

    class InitRandomizer:
        def __init__(self, rng):
            self.rng = rng
            self.ball_randomizer = DRConfig.BallRandomizer(rng)
            self.robot_randomizer = DRConfig.RobotRandomizer(rng)

        def init_robot(self, robot_id: int) -> tuple[float, float, float]:
            return self.robot_randomizer.randomize()

        def init_enemy_robot(self, robot_id: int) -> tuple[float, float, float]:
            return self.robot_randomizer.randomize()

        def init_ball(self) -> tuple[float, float]:
            return self.ball_randomizer.randomize()

    def __init__(self, seed=None):
        super().__init__()
        if seed is None:
            self.rng = random.default_rng()
        else:
            self.rng = random.default_rng(seed)
        self.init_randomizer = DRConfig.InitRandomizer(self.rng)


class TrueValueRandomizer(ObservationRandomizerBase):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def reseed(self) -> None:
        pass

    def get_seed(self) -> int:
        return 42

    def get_ball_error_xy(
        self, robot_vx: float, robot_vy: float, ball_rel_x: float, ball_rel_y: float
    ):
        return 0.0, 0.0

    def get_selfpos_error_xy(self):
        return 0.0, 0.0

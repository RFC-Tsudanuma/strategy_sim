import os
import sys
from typing import Optional

# Add the scripts directory to the path
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(scripts_dir)

from gymnasium import Env, spaces
from main import SimulationMain
from numpy import array, concatenate, float32, inf, ndarray, zeros
from randomize import ObservationRandomizerBase
from setting import ObjectLabel
from training_interface import (
    DomainRandomizer,
    GameInfo,
    InputNeckCommand,
    InputWalkCommand,
    RobotData,
)


def convert_np(data: RobotData, last_action: ndarray) -> ndarray:
    """
    RobotDataをnumpy配列に変換する
    odom_x,y,theta 3
    neck_yaw 1
    selfpos_x,y,theta 3
    object_list 2 + 4 + 4 + 6 = 16
        - ball x,y 2
        - goalpost x,y 2 * 2
        - person x,y 2 * 2
        - opponent x,y 2 * 3
        複数あるものは順不同で入れる。埋まらない所はマスクする
    合計 3 + 1 + 3 + 2 + 4 + 4 + 6 = 23
    マスクはobject_list用に使う
    総計 23 + 8 = 31
    """
    obs = zeros(23, dtype=float32)
    mask = zeros(8, dtype=float32)
    obs[0] = data.odom_x
    obs[1] = data.odom_y
    obs[2] = data.odom_theta
    obs[3] = data.neck_yaw
    obs[4] = data.selfpos_x
    obs[5] = data.selfpos_y
    obs[6] = data.selfpos_theta
    ball_count = 0
    goalpost_count = 0
    person_count = 0
    opponent_count = 0
    for obj in data.object_list:
        if obj.object_label == ObjectLabel.LABEL_BALL and ball_count < 1:
            obs[7] = obj.object_x
            obs[8] = obj.object_y
            mask[0] = 1.0
            ball_count += 1
        elif obj.object_label == ObjectLabel.LABEL_GOALPOST and goalpost_count < 2:
            obs[9 + goalpost_count * 2] = obj.object_x
            obs[10 + goalpost_count * 2] = obj.object_y
            mask[1 + goalpost_count] = 1.0
            goalpost_count += 1
        elif obj.object_label == ObjectLabel.LABEL_ROBOT and person_count < 2:
            obs[13 + person_count * 2] = obj.object_x
            obs[14 + person_count * 2] = obj.object_y
            mask[3 + person_count] = 1.0
            person_count += 1
        elif obj.object_label == ObjectLabel.LABEL_OPPENENT and opponent_count < 3:
            obs[17 + opponent_count * 2] = obj.object_x
            obs[18 + opponent_count * 2] = obj.object_y
            mask[5 + opponent_count] = 1.0
            opponent_count += 1
    return concatenate([obs, mask, last_action])


def convert_maskless_np(data: RobotData, last_action: ndarray) -> ndarray:
    """
    RobotDataをnumpy配列に変換する
    odom_x,y,theta 3
    neck_yaw 1
    selfpos_x,y,theta 3
    object_list 2 + 4 + 4 + 6 = 16
        - ball x,y 2
        - goalpost x,y 2 * 2
        - person x,y 2 * 2
        - opponent x,y 2 * 3
        オブジェクトは必ず全て入るのでマスクは無し。
    """
    obs = zeros(23, dtype=float32)
    obs[0] = data.odom_x
    obs[1] = data.odom_y
    obs[2] = data.odom_theta
    obs[3] = data.neck_yaw
    obs[4] = data.selfpos_x
    obs[5] = data.selfpos_y
    obs[6] = data.selfpos_theta
    ball_count = 0
    goalpost_count = 0
    person_count = 0
    opponent_count = 0
    for obj in data.object_list:
        if obj.object_label == ObjectLabel.LABEL_BALL and ball_count < 1:
            obs[7] = obj.object_x
            obs[8] = obj.object_y
            ball_count += 1
        elif obj.object_label == ObjectLabel.LABEL_GOALPOST and goalpost_count < 2:
            obs[9 + goalpost_count * 2] = obj.object_x
            obs[10 + goalpost_count * 2] = obj.object_y
            goalpost_count += 1
        elif obj.object_label == ObjectLabel.LABEL_ROBOT and person_count < 2:
            obs[13 + person_count * 2] = obj.object_x
            obs[14 + person_count * 2] = obj.object_y
            person_count += 1
        elif obj.object_label == ObjectLabel.LABEL_OPPENENT and opponent_count < 3:
            obs[17 + opponent_count * 2] = obj.object_x
            obs[18 + opponent_count * 2] = obj.object_y
            opponent_count += 1
    return concatenate([obs, last_action])


def convert_action_np_to_input(
    action: ndarray,
) -> tuple[InputWalkCommand, InputNeckCommand]:
    walk = InputWalkCommand()
    walk.x_velocity = action[0]
    walk.y_velocity = action[1]
    walk.theta_velocity = action[2]
    neck = InputNeckCommand()
    neck.neck_yaw_angle = action[3]
    neck.neck_pitch_angle = action[4]
    return walk, neck


def convert_action_np_to_input_without_neck(
    action: ndarray,
) -> InputWalkCommand:
    walk = InputWalkCommand()
    walk.x_velocity = action[0]
    walk.y_velocity = action[1]
    walk.theta_velocity = action[2]
    return walk


class StrategySimEnv(Env):
    def __init__(
        self,
        action_hz: int,
        target_robot: int = 1,
        enable_rendering: bool = False,
        max_episode_len: int = 5000,
        seed=None,
        randomizer: Optional[DomainRandomizer] = DomainRandomizer(),
        obs_randomizer: Optional[ObservationRandomizerBase] = None,
        complete_observation=False,
    ):
        super(StrategySimEnv, self).__init__()
        self.simulation = SimulationMain(
            ros2_enable=False,
            enable_drawing=enable_rendering,
            disable_print=True,
            observation_targets={target_robot},
            complete_observation=complete_observation,
            seed=seed,
            domain_randomizer=randomizer,
            obs_randomizer=obs_randomizer,
        )
        self.simulation.enable_goal_and_reset()  # ゴール後にリセットするように設定
        if enable_rendering:
            self.simulation.enable_rendering()
        if complete_observation:
            self.action_space = spaces.Box(
                low=array([-1.5, -1.5, -5.0]),
                high=array([1.5, 1.5, 5.0]),
                shape=(3,),
                dtype=float32,
            )
        else:
            self.action_space = spaces.Box(
                low=array([-1.5, -1.5, -5.0, -1.6, -1.6]),
                high=array([1.5, 1.5, 5.0, 1.6, 1.6]),
                shape=(5,),
                dtype=float32,
            )
        tmp_dt = RobotData()
        self.last_action: tuple[InputWalkCommand, InputNeckCommand] = (
            InputWalkCommand(),
            InputNeckCommand(),
        )
        self.last_action_np: ndarray = zeros(self.action_space.shape, dtype=float32)
        obs_shape = (
            convert_np(tmp_dt, self.last_action_np).shape
            if not complete_observation
            else convert_maskless_np(tmp_dt, self.last_action_np).shape
        )
        self.observation_space = spaces.Box(
            low=-inf, high=inf, shape=obs_shape, dtype=float32
        )
        self.target_robot = target_robot
        self.prev_game_state = None
        self.action_hz = action_hz
        self.simulation.reset()
        self.episode_len = 0
        self.max_episode_len = max_episode_len
        self.complete_observation = complete_observation

    def calc_reward(
        self, observation: list[RobotData], game_state: GameInfo
    ) -> tuple[float, dict]:
        # 報酬は0にしておく。継承先で定義する
        return 0.0, {}

    def convert_obs_to_np(self, data: RobotData) -> ndarray:
        if self.simulation.complete_observation:
            return convert_maskless_np(data, self.last_action_np)
        else:
            return convert_np(data, self.last_action_np)

    def reset(self, seed=None, options=None) -> tuple[ndarray, dict]:
        self.simulation.reset()
        self.simulation.step()  # 一度stepを実行してrobot_data_dictを初期化
        robot_data_list, gameinfo = self.simulation.get_observation()
        observation_np = self.convert_obs_to_np(robot_data_list[self.target_robot - 1])
        info = {}
        self.episode_len = 0
        return observation_np, info

    def step(self, action_np: ndarray) -> tuple[ndarray, float, bool, bool, dict]:
        self.last_action_np = action_np
        if self.complete_observation:
            walk_command = convert_action_np_to_input_without_neck(action_np)
            neck_command = InputNeckCommand()
            self.last_action = (walk_command, neck_command)
        else:
            walk_command, neck_command = convert_action_np_to_input(action_np)
            self.last_action = (walk_command, neck_command)
        self.simulation.set_command(self.target_robot, walk_command, neck_command)
        if self.simulation.sim_fps() == 10:
            self.simulation.step(obs_is_unnecessary=True)
            self.simulation.step(obs_is_unnecessary=True)
            self.simulation.step(obs_is_unnecessary=True)
            self.simulation.step(obs_is_unnecessary=True)
            self.simulation.step(obs_is_unnecessary=True)
            self.simulation.step(
                obs_is_unnecessary=True
            )  # 典型的な10fps環境用のloop unroll
        else:
            for _ in range(int(self.simulation.sim_fps() / self.action_hz)):
                self.simulation.step()
        raw_observation, gameinfo = self.simulation.get_observation()
        obs_np = self.convert_obs_to_np(raw_observation[self.target_robot - 1])
        info = {}
        reward, rew_info = self.calc_reward(raw_observation, gameinfo)
        info["reward_info"] = rew_info
        self.episode_len += 1
        # obs, reward, terminated, done, info
        return (
            obs_np,
            reward,
            self.terminated(raw_observation, gameinfo),
            self.done(),
            info,
        )

    def terminated(self, observation: list[RobotData], game_state: GameInfo) -> bool:
        return False

    def render(self, mode="") -> None:
        if mode == "human":
            self.enable_rendering()

    def done(self) -> bool:
        return self.episode_len > self.max_episode_len

    def close(self) -> None:
        self.simulation.finish_sim()

    def enable_rendering(self) -> None:
        self.simulation.enable_rendering()

    def current_step(self) -> int:
        return self.simulation.current_step

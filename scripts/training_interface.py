import copy
import dataclasses
import typing

import setting


@dataclasses.dataclass
class GameInfo:
    our_score: int = 0
    opponent_score: int = 0
    time_remaining: float = 0.0  # in seconds
    scored_team: setting.GoalType = setting.GoalType.NONE


@dataclasses.dataclass
class ObjectData:
    object_label: int = setting.ObjectLabel.LABEL_UNKNOWN
    object_x: float = 0.0
    object_y: float = 0.0


@dataclasses.dataclass
class RobotData:
    robot_id: int = -1
    odom_x: float = 0.0
    odom_y: float = 0.0
    odom_theta: float = 0.0
    neck_yaw: float = 0.0
    selfpos_x: float = 0.0
    selfpos_y: float = 0.0
    selfpos_theta: float = 0.0
    object_list: typing.List[ObjectData] = dataclasses.field(default_factory=list)


def pick_object(object_list: typing.List[ObjectData], label: setting.ObjectLabel):
    return [obj for obj in object_list if obj.object_label == label]


"""
ros2によるデータの送信を行わない場合に、データを返すために利用するダミー
"""


class DummyPublisher:
    def __init__(self, num_of_robots) -> None:
        self.num_of_robots = num_of_robots
        # robot_idは1-indexedなのでアクセスに注意すること
        self.robot_data_list: list[typing.Optional[RobotData]] = [
            None for i in range(num_of_robots)
        ]
        self.true_state_list: list[typing.Optional[RobotData]] = [
            None for i in range(num_of_robots)
        ]

    def create_elem(self, robot_id):
        if self.robot_data_list[robot_id - 1] is None:
            self.robot_data_list[robot_id - 1] = RobotData(robot_id=robot_id)

    def create_true_elem(self, robot_id):
        if self.true_state_list[robot_id - 1] is None:
            self.true_state_list[robot_id - 1] = RobotData(robot_id=robot_id)

    def pub_odometer(self, odom_x, odom_y, odom_theta, index):
        self.create_elem(index)
        self.create_true_elem(index)
        robot_data = self.robot_data_list[index - 1]
        robot_data.odom_x = odom_x
        robot_data.odom_y = odom_y
        robot_data.odom_theta = odom_theta
        true_robot_data = self.true_state_list[index - 1]
        true_robot_data.odom_x = odom_x
        true_robot_data.odom_y = odom_y
        true_robot_data.odom_theta = odom_theta

    def pub_neck_yaw(self, robot_neck_yaw, robot_neck_pitch, index):
        self.create_elem(index)
        robot_data = self.robot_data_list[index - 1]
        robot_data.neck_yaw = robot_neck_yaw
        robot_data.neck_pitch = robot_neck_pitch

    def pub_selfpos(self, robot_selfpos_x, robot_selfpos_y, robot_angle, target_robot):
        self.create_elem(target_robot)
        robot_data = self.robot_data_list[target_robot - 1]
        robot_data.selfpos_x = robot_selfpos_x
        robot_data.selfpos_y = robot_selfpos_y
        robot_data.selfpos_theta = robot_angle

    def pub_true_selfpos(
        self, robot_selfpos_x, robot_selfpos_y, robot_angle, target_robot
    ):
        self.create_true_elem(target_robot)
        true_robot_data = self.true_state_list[target_robot - 1]
        true_robot_data.selfpos_x = robot_selfpos_x
        true_robot_data.selfpos_y = robot_selfpos_y
        true_robot_data.selfpos_theta = robot_angle

    def add_object_pos(self, OBJECT_LABEL, object_x, object_y, target_robot):
        self.create_elem(target_robot)
        robot_data = self.robot_data_list[target_robot - 1]
        robot_data.object_list.append(
            ObjectData(object_label=OBJECT_LABEL, object_x=object_x, object_y=object_y)
        )

    def add_true_object_pos(self, OBJECT_LABEL, object_x, object_y, target_robot):
        self.create_true_elem(target_robot)
        robot_data = self.true_state_list[target_robot - 1]
        robot_data.object_list.append(
            ObjectData(object_label=OBJECT_LABEL, object_x=object_x, object_y=object_y)
        )

    def publish_object_pos(self):
        # これは実装しなくてよい
        pass

    def publish_true_object_pos(self):
        # これは実装しなくてよい
        pass

    def destroy_node(self):
        # これは実装しなくてよい
        pass

    def return_world_data(self):
        """
        add関数でまとめた世界の状態を1つのクラスでまとめて返す関数
        リストが返ってくる。1-indexedなrobot_idに対して-1してアクセスすること
        """
        # データが存在する要素のみをdeep copyして返す
        result = []
        for robot_data in self.robot_data_list:
            if robot_data is not None:
                # 必要な部分のみをコピー
                copied_robot = RobotData(
                    robot_id=robot_data.robot_id,
                    odom_x=robot_data.odom_x,
                    odom_y=robot_data.odom_y,
                    odom_theta=robot_data.odom_theta,
                    neck_yaw=robot_data.neck_yaw,
                    selfpos_x=robot_data.selfpos_x,
                    selfpos_y=robot_data.selfpos_y,
                    selfpos_theta=robot_data.selfpos_theta,
                    object_list=robot_data.object_list.copy(),  # shallow copy of list
                )
                result.append(copied_robot)
            else:
                result.append(None)

        # リセット（object_listもクリア）
        for robot_data in self.robot_data_list:
            if robot_data is not None:
                robot_data.object_list.clear()

        for robot_data in self.true_state_list:
            if robot_data is not None:
                robot_data.object_list.clear()

        return result


class DummyInputHandler:
    def __init__(self) -> None:
        self.dragging_ball = False  # 常にfalseとする
        self.dragging_robot = 765  # 常に何にも当てはまらない数字にする

    def handle_events(self, events, ball, robots, enemy_robots):
        # これは実装する必要が無い
        return None


"""
このクラスは本来のコマンドと同じく[m/s]で指定する
"""


@dataclasses.dataclass
class InputWalkCommand:
    x_velocity: float = 0.0
    y_velocity: float = 0.0
    theta_velocity: float = 0.0


@dataclasses.dataclass
class InputNeckCommand:
    neck_yaw_angle: float = 0.0
    neck_pitch_angle: float = 0.0


class DummySubscriber:
    def __init__(self, is_enemy=False) -> None:
        self.walk_command_dict: typing.Dict[int, InputWalkCommand] = {}
        self.neck_command_dict: typing.Dict[int, InputNeckCommand] = {}
        self.is_enemy = is_enemy

    """
    1-indexed
    敵の場合は4,5,6
    """

    def set_walk_command(self, command: InputWalkCommand, index):
        self.walk_command_dict[index] = command

    """
    1-indexed
    敵の場合は4,5,6
    """

    def set_neck_command(self, command: InputNeckCommand, index):
        self.neck_command_dict[index] = command

    def get_latest_command_all(self):
        result = [None, None, None]
        for key, value in self.walk_command_dict.items():
            if not self.is_enemy:
                result[key - 1] = value
            else:
                result[key - 4] = value
        self.walk_command_dict.clear()
        return result[0], result[1], result[2]

    def get_latest_neck_command_all(self):
        result = [None, None, None]
        for key, value in self.neck_command_dict.items():
            if not self.is_enemy:
                result[key - 1] = value
            else:
                result[key - 4] = value

        self.neck_command_dict.clear()
        return result[0], result[1], result[2]

    def destroy_node(self):
        # これは実装しなくてよい
        pass


class DomainRandomizer:
    def __init__(self):
        self.init_randomizer = None
        pass

    def init_ball(self) -> tuple[float, float]:
        if self.init_randomizer is None:
            return None, None
        else:
            return self.init_randomizer.init_ball()

    def init_robot(self, robot_id: int) -> tuple[float, float, float]:
        if self.init_randomizer is None:
            return 0.0, 0.0, 0.0
        else:
            return self.init_randomizer.init_robot(robot_id)

    def init_enemy_robot(self, robot_id: int) -> tuple[float, float, float]:
        if self.init_randomizer is None:
            return 0.0, 0.0, 0.0
        else:
            return self.init_randomizer.init_enemy_robot(robot_id)

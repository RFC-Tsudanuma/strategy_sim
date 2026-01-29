import json
import threading

import rclpy
from booster_msgs.msg import RpcReqMsg
from rclpy.node import Node


class WalkCommand:
    x_velocity: float
    y_velocity: float
    theta_velocity: float


class NeckCommand:
    neck_yaw_angle: float
    neck_pitch_angle: float


class WalkCommandSubscriber(Node):
    def __init__(self, is_enemy=False):
        super().__init__("strategy_sim_walk_command")
        self.is_enemy = is_enemy
        if not self.is_enemy:
            self.subscription_robot_1 = self.create_subscription(
                RpcReqMsg, "LocoApiTopicReq_1", self.listener_callback_1, 1
            )
            self.subscription_robot_2 = self.create_subscription(
                RpcReqMsg, "LocoApiTopicReq_2", self.listener_callback_2, 1
            )
            self.subscription_robot_3 = self.create_subscription(
                RpcReqMsg, "LocoApiTopicReq_3", self.listener_callback_3, 1
            )
        else:
            self.subscription_robot_1 = self.create_subscription(
                RpcReqMsg, "LocoApiTopicReq_4", self.listener_callback_1, 1
            )
            self.subscription_robot_2 = self.create_subscription(
                RpcReqMsg, "LocoApiTopicReq_5", self.listener_callback_2, 1
            )
            self.subscription_robot_3 = self.create_subscription(
                RpcReqMsg, "LocoApiTopicReq_6", self.listener_callback_3, 1
            )

        self.subscription_robot_1  # prevent unused variable warning
        self.subscription_robot_2  # prevent unused variable warning
        self.subscription_robot_3  # prevent unused variable warning
        self.walk_command_lock = threading.Lock()
        self.neck_command_lock = threading.Lock()
        self.walk_command_latest = [None, None, None]
        self.neck_command_latest = [None, None, None]

    def listener_callback_1(self, msg):
        if "2001" in msg.header:  # 歩行コマンドだけを受け取る
            json_msg = json.loads(msg.body)
            cmd = WalkCommand()
            cmd.x_velocity = json_msg["vx"]
            cmd.y_velocity = -json_msg["vy"]  # y軸の符号を反転。実ロボットは右手系。
            cmd.theta_velocity = json_msg["vyaw"]
            with self.walk_command_lock:
                self.walk_command_latest[0] = cmd
            if not self.is_enemy:
                print(
                    f"Received command for robot 1: {cmd.x_velocity}, {cmd.y_velocity}, {cmd.theta_velocity}"
                )
            else:
                print(
                    f"Received command for enemy robot 4: {cmd.x_velocity}, {cmd.y_velocity}, {cmd.theta_velocity}"
                )
        if "2004" in msg.header:
            cmd = NeckCommand()
            json_msg = json.loads(msg.body)
            cmd.neck_yaw_angle = -json_msg["yaw"]
            cmd.neck_pitch_angle = json_msg["pitch"]
            with self.neck_command_lock:
                self.neck_command_latest[0] = cmd
            if not self.is_enemy:
                print(
                    f"Received neck command for robot 1: {cmd.neck_yaw_angle}, {cmd.neck_pitch_angle}"
                )
            else:
                print(
                    f"Received neck command for enemy robot 4: {cmd.neck_yaw_angle}, {cmd.neck_pitch_angle}"
                )

    def listener_callback_2(self, msg):
        if "2001" in msg.header:  # 歩行コマンドだけを受け取る
            json_msg = json.loads(msg.body)
            cmd = WalkCommand()
            cmd.x_velocity = json_msg["vx"]
            cmd.y_velocity = -json_msg["vy"]  # y軸の符号を反転。実ロボットは右手系。
            cmd.theta_velocity = json_msg["vyaw"]
            with self.walk_command_lock:
                self.walk_command_latest[1] = cmd
            if not self.is_enemy:
                print(
                    f"Received command for robot 2: {cmd.x_velocity}, {cmd.y_velocity}, {cmd.theta_velocity}"
                )
            else:
                print(
                    f"Received command for enemy robot 5: {cmd.x_velocity}, {cmd.y_velocity}, {cmd.theta_velocity}"
                )
        if "2004" in msg.header:
            cmd = NeckCommand()
            json_msg = json.loads(msg.body)
            cmd.neck_yaw_angle = -json_msg["yaw"]
            cmd.neck_pitch_angle = json_msg["pitch"]
            with self.neck_command_lock:
                self.neck_command_latest[1] = cmd
            if not self.is_enemy:
                print(
                    f"Received neck command for robot 2: {cmd.neck_yaw_angle}, {cmd.neck_pitch_angle}"
                )
            else:
                print(
                    f"Received neck command for enemy robot 5: {cmd.neck_yaw_angle}, {cmd.neck_pitch_angle}"
                )

    def listener_callback_3(self, msg):
        if "2001" in msg.header:  # 歩行コマンドだけを受け取る
            json_msg = json.loads(msg.body)
            cmd = WalkCommand()
            cmd.x_velocity = json_msg["vx"]
            cmd.y_velocity = -json_msg["vy"]  # y軸の符号を反転。実ロボットは右手系。
            cmd.theta_velocity = json_msg["vyaw"]
            with self.walk_command_lock:
                self.walk_command_latest[2] = cmd
            if not self.is_enemy:
                print(
                    f"Received command for robot 3: {cmd.x_velocity}, {cmd.y_velocity}, {cmd.theta_velocity}"
                )
            else:
                print(
                    f"Received command for enemy robot 6: {cmd.x_velocity}, {cmd.y_velocity}, {cmd.theta_velocity}"
                )
        if "2004" in msg.header:
            cmd = NeckCommand()
            json_msg = json.loads(msg.body)
            cmd.neck_yaw_angle = -json_msg["yaw"]
            cmd.neck_pitch_angle = json_msg["pitch"]
            with self.neck_command_lock:
                self.neck_command_latest[2] = cmd
            if not self.is_enemy:
                print(
                    f"Received neck command for robot 3: {cmd.neck_yaw_angle}, {cmd.neck_pitch_angle}"
                )
            else:
                print(
                    f"Received neck command for enemy robot 6: {cmd.neck_yaw_angle}, {cmd.neck_pitch_angle}"
                )

    def get_latest_command_all(self):
        with self.walk_command_lock:
            cmd1 = self.walk_command_latest[0]
            cmd2 = self.walk_command_latest[1]
            cmd3 = self.walk_command_latest[2]

            self.walk_command_latest[0] = None
            self.walk_command_latest[1] = None
            self.walk_command_latest[2] = None
            return cmd1, cmd2, cmd3

    def get_latest_neck_command_all(self):
        with self.neck_command_lock:
            cmd1 = self.neck_command_latest[0]
            cmd2 = self.neck_command_latest[1]
            cmd3 = self.neck_command_latest[2]

            self.neck_command_latest[0] = None
            self.neck_command_latest[1] = None
            self.neck_command_latest[2] = None
            return cmd1, cmd2, cmd3

#!/usr/bin/env python3
import math
import threading
import time
import typing

import field
import numpy as np
import pygame
import setting
import training_interface
from ball import Ball
from collision import check_robot_ball_collision, check_robot_robot_collision
from draw_randomized_ball import draw_randomized_ball
from enemy_robot import EnemyRobot
from field_calc import calc_robot_and_object_relative_position, convert_to_robot_coord
from game_state import GameState
from input_handler import InputHandler
from randomize import ObservationRandomizerBase, Randomize
from robot import Robot
from setting import DETECTION_DISTANCE_SCALE_X, DETECTION_DISTANCE_SCALE_Y, ObjectLabel
from training_interface import DomainRandomizer, InputNeckCommand, InputWalkCommand


class SimQuitException(Exception):
    pass


class SimulationMain:
    def __init__(
        self,
        ros2_enable=True,
        enable_drawing=True,
        disable_print=False,
        observation_targets: set[int] = set(setting.ROBOTS_ID_LIST),
        seed: typing.Optional[int] = None,
        complete_observation: bool = False,
        domain_randomizer: typing.Optional[DomainRandomizer] = DomainRandomizer(),
        obs_randomizer: typing.Optional[ObservationRandomizerBase] = None,
    ):
        """
        Args:
            ros2_enable (bool): ROS2を有効にするかどうか
            enable_drawing (bool): pygameによる描画を有効にするかどうか
            disable_print (bool): ロボットの状態表示を無効にするかどうか
            observation_targets (set[int]): 観測対象のロボット番号の集合（1-indexed）これに無いものは観測として返されない。
                                            デフォルトは全てのロボットが対象。強化学習などの場合に特定のロボットのみ観測したい場合に使用する。
        Returns:
            None
        """
        self.ros2_enable = ros2_enable
        self.enable_drawing = enable_drawing
        self.screen = None
        self.clock = None
        self.disable_print = disable_print
        self.game_state = GameState()
        self.prev_time = time.time()
        self.goal_and_reset_enabled = False
        self.observation_targets = observation_targets
        if obs_randomizer is None:
            self.randomize = Randomize(seed=seed)
        else:
            self.randomize = obs_randomizer
        self.complete_observation = complete_observation
        self.domain_randomizer = domain_randomizer

        if self.ros2_enable:
            import rclpy

            rclpy.init()

        if self.enable_drawing:
            self.initialize_drawing()
        else:
            self.input_handler = training_interface.DummyInputHandler()

        self.initialize_objects()

        if ros2_enable:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor

            from strategy_sim.publisher_member_function import WorldStatePublisher
            from strategy_sim.subscriber_member_function import WalkCommandSubscriber

            self.world_state_publisher = WorldStatePublisher()
            self.walk_command_subscriber = WalkCommandSubscriber()
            self.enemy_walk_command_subscriber = WalkCommandSubscriber(is_enemy=True)
            self.exec = SingleThreadedExecutor()
            self.exec.add_node(self.world_state_publisher)
            self.exec.add_node(self.walk_command_subscriber)
            self.exec.add_node(self.enemy_walk_command_subscriber)
            self.sub_thread = threading.Thread(target=self.exec.spin)
            self.sub_thread.daemon = True
            self.sub_thread.start()
        else:
            self.world_state_publisher = training_interface.DummyPublisher(
                num_of_robots=setting.NUM_OF_ROBOTS
            )
            self.walk_command_subscriber = training_interface.DummySubscriber()
            self.enemy_walk_command_subscriber = training_interface.DummySubscriber(
                is_enemy=True
            )

    def initialize_objects(self):
        # ボールの初期化
        x, y = self.domain_randomizer.init_ball()
        self.ball = Ball(x=x, y=y)
        # 3台のロボットを生成（それぞれ異なる初期位置と速度）
        x, y, w = self.domain_randomizer.init_robot(1)
        x2, y2, w2 = self.domain_randomizer.init_robot(2)
        x3, y3, w3 = self.domain_randomizer.init_robot(3)
        self.robots = [
            Robot(
                robot_number=1,
                x=100 + x,
                y=setting.HEIGHT + y,
                angle=math.pi / 2 + w,
                complete_observation=self.complete_observation,
            ),
            Robot(
                robot_number=2,
                x=300 + x2,
                y=setting.HEIGHT + y2,
                angle=math.pi / 2 + w2,
                complete_observation=self.complete_observation,
            ),
            Robot(
                robot_number=3,
                x=500 + x3,
                y=setting.HEIGHT + y3,
                angle=math.pi / 2 + w3,
                complete_observation=self.complete_observation,
            ),
        ]
        # 3台の敵ロボットを生成（フィールドの右側に配置）
        x, y, w = self.domain_randomizer.init_robot(4)
        x2, y2, w2 = self.domain_randomizer.init_robot(5)
        x3, y3, w3 = self.domain_randomizer.init_robot(6)
        self.enemy_robots = [
            EnemyRobot(
                robot_number=1,
                x=3 * setting.WIDTH // 4 + x,
                y=setting.HEIGHT // 3 + y,
                angle=math.pi + w,
                complete_observation=self.complete_observation,
            ),
            EnemyRobot(
                robot_number=2,
                x=3 * setting.WIDTH // 4 + x2,
                y=setting.HEIGHT // 2 + y2,
                angle=math.pi + w2,
                complete_observation=self.complete_observation,
            ),
            EnemyRobot(
                robot_number=3,
                x=3 * setting.WIDTH // 4 + x3,
                y=2 * setting.HEIGHT // 3 + y3,
                angle=math.pi + w3,
                complete_observation=self.complete_observation,
            ),
        ]

    # 初期化
    def step(self, obs_is_unnecessary=False) -> None:
        if self.enable_drawing:
            # イベント取得
            events = pygame.event.get()

            # 終了チェック
            for event in events:
                if event.type == pygame.QUIT:
                    raise SimQuitException("Quit event detected!!!")
        else:
            events = None

        # 入力処理（味方ロボットと敵ロボットの両方を含める）
        all_robots = self.robots + self.enemy_robots
        _ = self.input_handler.handle_events(
            events, self.ball, self.robots, self.enemy_robots
        )
        r_cmd1, r_cmd2, r_cmd3 = self.walk_command_subscriber.get_latest_command_all()
        if r_cmd1 is not None:
            self.robots[0].set_target_velocity(
                r_cmd1.x_velocity, r_cmd1.y_velocity, r_cmd1.theta_velocity
            )
        if r_cmd2 is not None:
            self.robots[1].set_target_velocity(
                r_cmd2.x_velocity, r_cmd2.y_velocity, r_cmd2.theta_velocity
            )
        if r_cmd3 is not None:
            self.robots[2].set_target_velocity(
                r_cmd3.x_velocity, r_cmd3.y_velocity, r_cmd3.theta_velocity
            )
        er_cmd1, er_cmd2, er_cmd3 = (
            self.enemy_walk_command_subscriber.get_latest_command_all()
        )
        if er_cmd1 is not None:
            self.enemy_robots[0].set_target_velocity(
                er_cmd1.x_velocity, er_cmd1.y_velocity, er_cmd1.theta_velocity
            )
        if er_cmd2 is not None:
            self.enemy_robots[1].set_target_velocity(
                er_cmd2.x_velocity, er_cmd2.y_velocity, er_cmd2.theta_velocity
            )
        if er_cmd3 is not None:
            self.enemy_robots[2].set_target_velocity(
                er_cmd3.x_velocity, er_cmd3.y_velocity, er_cmd3.theta_velocity
            )

        n_cmd1, n_cmd2, n_cmd3 = (
            self.walk_command_subscriber.get_latest_neck_command_all()
        )
        if n_cmd1 is not None:
            self.robots[0].set_neck_yaw(n_cmd1.neck_yaw_angle)
            self.robots[0].set_neck_pitch(n_cmd1.neck_pitch_angle)
        if n_cmd2 is not None:
            self.robots[1].set_neck_yaw(n_cmd2.neck_yaw_angle)
            self.robots[1].set_neck_pitch(n_cmd2.neck_pitch_angle)
        if n_cmd3 is not None:
            self.robots[2].set_neck_yaw(n_cmd3.neck_yaw_angle)
            self.robots[2].set_neck_pitch(n_cmd3.neck_pitch_angle)
        nn_cmd1, nn_cmd2, nn_cmd3 = (
            self.enemy_walk_command_subscriber.get_latest_neck_command_all()
        )
        if nn_cmd1 is not None:
            self.enemy_robots[0].set_neck_yaw(nn_cmd1.neck_yaw_angle)
            self.enemy_robots[0].set_neck_pitch(nn_cmd1.neck_pitch_angle)
        if nn_cmd2 is not None:
            self.enemy_robots[1].set_neck_yaw(nn_cmd2.neck_yaw_angle)
            self.enemy_robots[1].set_neck_pitch(nn_cmd2.neck_pitch_angle)
        if nn_cmd3 is not None:
            self.enemy_robots[2].set_neck_yaw(nn_cmd3.neck_yaw_angle)
            self.enemy_robots[2].set_neck_pitch(nn_cmd3.neck_pitch_angle)

        # 更新処理
        # ボールの更新（ドラッグ中でない場合のみ）
        if not self.input_handler.dragging_ball:
            self.ball.update()

        # ロボットの更新
        for i, robot in enumerate(self.robots):
            # ドラッグ中のロボット以外のみ更新
            if self.input_handler.dragging_robot != i:
                robot.update(disable_print=self.disable_print)

        # 敵ロボットの更新
        for i, robot in enumerate(self.enemy_robots):
            # ドラッグ中のロボット以外のみ更新（味方ロボットの数を考慮）
            if self.input_handler.dragging_robot != i + len(self.robots):
                robot.update(disable_print=self.disable_print)

        # 衝突判定
        # 全ロボット（味方と敵）のリストを作成
        all_robots_for_collision = self.robots + self.enemy_robots

        # ロボット同士の衝突判定（味方同士、敵同士、味方と敵の全ての組み合わせ）
        for i in range(len(all_robots_for_collision)):
            for j in range(i + 1, len(all_robots_for_collision)):
                check_robot_robot_collision(
                    all_robots_for_collision[i], all_robots_for_collision[j]
                )

        # ロボットとボールの衝突判定（味方と敵の両方）
        for robot in self.robots:
            check_robot_ball_collision(robot, self.ball)
        for robot in self.enemy_robots:
            check_robot_ball_collision(robot, self.ball)

        # ゴール判定と点数の更新
        goal_scored = self.ball.check_ball_in_goal()
        self.game_state.update_score(goal_scored)
        if self.goal_and_reset_enabled and goal_scored != setting.GoalType.NONE:
            self.ball.reset()
            for robot in self.robots:
                robot.reset_position_and_velocity()
            for robot in self.enemy_robots:
                robot.reset_position_and_velocity()

        # 関連情報をパブリッシュ
        index = 1
        randomized_balls = []  # 誤差を含んだボール位置を保存
        for robot in all_robots:
            # 観測対象でないロボットはスキップ
            if index not in self.observation_targets or obs_is_unnecessary:
                index += 1
                continue
            odom_x, odom_y, odom_theta = robot.get_odom()
            self.world_state_publisher.pub_odometer(odom_x, odom_y, odom_theta, index)
            self.world_state_publisher.pub_neck_yaw(
                robot.neck_yaw, robot.neck_pitch, index
            )
            selfpos_x, selfpos_y = convert_to_robot_coord(robot.x, robot.y)
            selfpos_err_x, selfpos_err_y = self.randomize.get_selfpos_error_xy()
            self.world_state_publisher.pub_selfpos(
                selfpos_x + selfpos_err_x,
                selfpos_y + selfpos_err_y,
                robot.angle,
                target_robot=index,
            )
            # 真の自己位置もpublish
            self.world_state_publisher.pub_true_selfpos(
                selfpos_x,
                selfpos_y,
                robot.angle,
                target_robot=index,
            )
            # ボールが視野内にある場合のみpublish
            if robot.is_in_view(self.ball.x, self.ball.y):
                x, y = calc_robot_and_object_relative_position(
                    robot.x,
                    robot.y,
                    robot.angle,
                    self.ball.x,
                    self.ball.y,
                    scale_x=DETECTION_DISTANCE_SCALE_X,
                    scale_y=DETECTION_DISTANCE_SCALE_Y,
                )
                error_x, error_y = self.randomize.get_ball_error_xy(
                    robot.vx, robot.vy, x, y
                )
                randomized_x = x + error_x
                randomized_y = y + error_y
                # エラーを含んだボールの位置をpublish
                self.world_state_publisher.add_object_pos(
                    ObjectLabel.LABEL_BALL,
                    randomized_x,
                    randomized_y,
                    target_robot=index,
                )
                # 真のボール位置もpublish
                self.world_state_publisher.add_true_object_pos(
                    ObjectLabel.LABEL_BALL,
                    x,
                    y,
                    target_robot=index,
                )
                # 誤差を含んだボール位置を画面座標に変換して保存
                # ロボット座標系からワールド座標系への変換
                rotation_mat = np.array(
                    [
                        [np.cos(-robot.angle), -np.sin(-robot.angle)],
                        [np.sin(-robot.angle), np.cos(-robot.angle)],
                    ]
                )
                rel_pos = np.array([randomized_x, -randomized_y])  # y軸の符号を戻す
                world_rel = rotation_mat @ rel_pos
                screen_x = robot.x + world_rel[0]
                screen_y = robot.y + world_rel[1]
                randomized_balls.append((index, screen_x, screen_y))
            # ゴールポスト（左上）が視野内にある場合のみpublish
            if robot.is_in_view(
                setting.GOALPOST_LEFT_UP[0], setting.GOALPOST_LEFT_UP[1]
            ):
                x, y = calc_robot_and_object_relative_position(
                    robot.x,
                    robot.y,
                    robot.angle,
                    setting.GOALPOST_LEFT_UP[0],
                    setting.GOALPOST_LEFT_UP[1],
                    scale_x=DETECTION_DISTANCE_SCALE_X,
                    scale_y=DETECTION_DISTANCE_SCALE_Y,
                )
                self.world_state_publisher.add_object_pos(
                    ObjectLabel.LABEL_GOALPOST, x, y, target_robot=index
                )
                self.world_state_publisher.add_true_object_pos(
                    ObjectLabel.LABEL_GOALPOST, x, y, target_robot=index
                )
            # ゴールポスト（左下）が視野内にある場合のみpublish
            if robot.is_in_view(
                setting.GOALPOST_LEFT_DOWN[0], setting.GOALPOST_LEFT_DOWN[1]
            ):
                x, y = calc_robot_and_object_relative_position(
                    robot.x,
                    robot.y,
                    robot.angle,
                    setting.GOALPOST_LEFT_DOWN[0],
                    setting.GOALPOST_LEFT_DOWN[1],
                    scale_x=DETECTION_DISTANCE_SCALE_X,
                    scale_y=DETECTION_DISTANCE_SCALE_Y,
                )
                self.world_state_publisher.add_object_pos(
                    ObjectLabel.LABEL_GOALPOST, x, y, target_robot=index
                )
                self.world_state_publisher.add_true_object_pos(
                    ObjectLabel.LABEL_GOALPOST, x, y, target_robot=index
                )
            # ゴールポスト（右上）が視野内にある場合のみpublish
            if robot.is_in_view(
                setting.GOALPOST_RIGHT_UP[0], setting.GOALPOST_RIGHT_UP[1]
            ):
                x, y = calc_robot_and_object_relative_position(
                    robot.x,
                    robot.y,
                    robot.angle,
                    setting.GOALPOST_RIGHT_UP[0],
                    setting.GOALPOST_RIGHT_UP[1],
                    scale_x=DETECTION_DISTANCE_SCALE_X,
                    scale_y=DETECTION_DISTANCE_SCALE_Y,
                )
                self.world_state_publisher.add_object_pos(
                    ObjectLabel.LABEL_GOALPOST, x, y, target_robot=index
                )
                self.world_state_publisher.add_true_object_pos(
                    ObjectLabel.LABEL_GOALPOST, x, y, target_robot=index
                )
            # ゴールポスト（右下）が視野内にある場合のみpublish
            if robot.is_in_view(
                setting.GOALPOST_RIGHT_DOWN[0], setting.GOALPOST_RIGHT_DOWN[1]
            ):
                x, y = calc_robot_and_object_relative_position(
                    robot.x,
                    robot.y,
                    robot.angle,
                    setting.GOALPOST_RIGHT_DOWN[0],
                    setting.GOALPOST_RIGHT_DOWN[1],
                    scale_x=DETECTION_DISTANCE_SCALE_X,
                    scale_y=DETECTION_DISTANCE_SCALE_Y,
                )
                self.world_state_publisher.add_object_pos(
                    ObjectLabel.LABEL_GOALPOST, x, y, target_robot=index
                )
                self.world_state_publisher.add_true_object_pos(
                    ObjectLabel.LABEL_GOALPOST, x, y, target_robot=index
                )
            ii = 1
            for tmp_robot in self.robots:
                if ii != index:
                    # 他のロボットが視野内にある場合のみpublish
                    if robot.is_in_view(tmp_robot.x, tmp_robot.y):
                        x, y = calc_robot_and_object_relative_position(
                            robot.x,
                            robot.y,
                            robot.angle,
                            tmp_robot.x,
                            tmp_robot.y,
                            scale_x=DETECTION_DISTANCE_SCALE_X,
                            scale_y=DETECTION_DISTANCE_SCALE_Y,
                        )
                        self.world_state_publisher.add_object_pos(
                            ObjectLabel.LABEL_ROBOT, x, y, target_robot=index
                        )
                        self.world_state_publisher.add_true_object_pos(
                            ObjectLabel.LABEL_ROBOT, x, y, target_robot=index
                        )
                ii += 1
            # 敵ロボットの位置も相対位置として追加（視野内のみ）
            for enemy in self.enemy_robots:
                if robot.is_in_view(enemy.x, enemy.y):
                    x, y = calc_robot_and_object_relative_position(
                        robot.x,
                        robot.y,
                        robot.angle,
                        enemy.x,
                        enemy.y,
                        scale_x=DETECTION_DISTANCE_SCALE_X,
                        scale_y=DETECTION_DISTANCE_SCALE_Y,
                    )
                    self.world_state_publisher.add_object_pos(
                        ObjectLabel.LABEL_OPPENENT, x, y, target_robot=index
                    )
                    self.world_state_publisher.add_true_object_pos(
                        ObjectLabel.LABEL_OPPENENT, x, y, target_robot=index
                    )
            # 自己位置推定のためにランドマークの相対位置を検出
            for lm in robot.get_detected_field_landmarks(setting.LANDMARK_TCROSS):
                error_x, error_y = self.randomize.get_ball_error_xy(
                    robot.vx, robot.vy, lm[0], lm[1]
                )
                self.world_state_publisher.add_object_pos(
                    ObjectLabel.LABEL_TCROSS,
                    lm[0] + error_x,
                    lm[1] + error_y,
                    target_robot=index,
                )
            for lm in robot.get_detected_field_landmarks(setting.LANDMARK_LCROSS):
                error_x, error_y = self.randomize.get_ball_error_xy(
                    robot.vx, robot.vy, lm[0], lm[1]
                )
                self.world_state_publisher.add_object_pos(
                    ObjectLabel.LABEL_LCROSS,
                    lm[0] + error_x,
                    lm[1] + error_y,
                    target_robot=index,
                )
            for lm in robot.get_detected_field_landmarks(setting.LANDMARK_XCROSS):
                error_x, error_y = self.randomize.get_ball_error_xy(
                    robot.vx, robot.vy, lm[0], lm[1]
                )
                self.world_state_publisher.add_object_pos(
                    ObjectLabel.LABEL_XCROSS,
                    lm[0] + error_x,
                    lm[1] + error_y,
                    target_robot=index,
                )
            for lm in robot.get_detected_field_landmarks(setting.LANDMARK_PENALTYPOINT):
                error_x, error_y = self.randomize.get_ball_error_xy(
                    robot.vx, robot.vy, lm[0], lm[1]
                )
                self.world_state_publisher.add_object_pos(
                    ObjectLabel.LABEL_PENALTYPOINT,
                    lm[0] + error_x,
                    lm[1] + error_y,
                    target_robot=index,
                )
            index += 1
        self.world_state_publisher.publish_object_pos()
        self.world_state_publisher.publish_true_object_pos()

        # 描画とスリープと時間更新
        if self.enable_drawing:
            # 時間の更新
            current_time = time.time()
            elapsed_time = current_time - self.prev_time
            self.prev_time = current_time
            self.game_state.update_time(elapsed_time)

            # 描画
            field.draw_field(self.screen)
            self.ball.draw(self.screen)
            for robot in self.robots:
                robot.draw(self.screen)
            for robot in self.enemy_robots:
                robot.draw(self.screen)
            self.game_state.draw_state(self.screen)  # 時間とスコアの描画

            # 誤差を含んだボールを最後に描画（他の要素に上書きされないように）
            for robot_idx, ball_x, ball_y in randomized_balls:
                draw_randomized_ball(self.screen, robot_idx, ball_x, ball_y)

            pygame.display.flip()
            self.clock.tick(setting.SIM_FPS)  # 60 FPS用のスリープ
        else:
            # 描画しない場合は1フレーム(1/fps)秒進んだことにする
            self.game_state.update_time(1.0 / setting.SIM_FPS)

    def initialize_drawing(self) -> None:
        if self.enable_drawing:
            pygame.init()
            self.screen = pygame.display.set_mode((setting.WIDTH, setting.HEIGHT))
            self.clock = pygame.time.Clock()
            self.input_handler = InputHandler()

    def finish_sim(self) -> None:
        print("Finish simulation...")
        if self.enable_drawing:
            pygame.quit()
        if self.ros2_enable:
            import rclpy

            self.exec.shutdown()
            self.world_state_publisher.destroy_node()
            self.walk_command_subscriber.destroy_node()
            self.enemy_walk_command_subscriber.destroy_node()
            rclpy.shutdown()
            self.sub_thread.join()

    """
    外部からロボットへのコマンドを設定する関数
    この関数はスレッドセーフではない
    target_robotは1-indexed
    """

    def set_command(
        self,
        target_robot: int,
        walk_command: typing.Optional[InputWalkCommand] = None,
        neck_command: typing.Optional[InputNeckCommand] = None,
    ) -> None:
        if walk_command is not None:
            if target_robot > 3:
                self.enemy_walk_command_subscriber.set_walk_command(
                    walk_command, target_robot
                )
            else:
                self.walk_command_subscriber.set_walk_command(
                    walk_command, target_robot
                )
        if neck_command is not None:
            if target_robot > 3:
                self.enemy_walk_command_subscriber.set_neck_command(
                    neck_command, target_robot
                )
            else:
                self.walk_command_subscriber.set_neck_command(
                    neck_command, target_robot
                )

    def get_observation(
        self,
    ) -> typing.Tuple[list[training_interface.RobotData], training_interface.GameInfo]:
        if self.ros2_enable:
            raise RuntimeError(
                "get_observation is not available when ros2_enable is True"
            )
        return (
            self.world_state_publisher.return_world_data(),
            self.game_state.to_game_info(),
        )

    def reset(self) -> None:
        """シミュレーションを初期状態にリセットする関数"""
        self.game_state.reset()
        self.initialize_objects()

    def sim_fps(self) -> int:
        return setting.SIM_FPS

    def enable_rendering(self) -> None:
        if not self.enable_drawing:
            self.enable_drawing = True
            self.initialize_drawing()

    def enable_goal_and_reset(self) -> None:
        self.goal_and_reset_enabled = True


if __name__ == "__main__":
    sim = SimulationMain(ros2_enable=True, enable_drawing=True, disable_print=False)
    try:
        while True:
            sim.step()
    except SimQuitException as e:
        print(e)
        sim.finish_sim()

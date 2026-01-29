import math

import pygame
import setting


class InputHandler:
    def __init__(self):
        self.dragging_ball = False
        self.dragging_robot = None  # 左クリックでドラッグ中のロボットのインデックス
        self.rotating_robot = None  # 右クリックで回転中のロボットのインデックス

    def handle_events(self, events, ball, robots=[], enemy_robots=[]):
        """イベントを処理し、必要に応じてボールやロボットの状態を更新"""
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                # まずロボットの近くかチェック（味方と敵の両方）
                all_robots = robots + enemy_robots
                for i, robot in enumerate(all_robots):
                    distance = math.sqrt(
                        (robot.x - mouse_x) ** 2 + (robot.y - mouse_y) ** 2
                    )
                    if (
                        distance < robot.size // 2 + 10
                    ):  # ロボットの近くでクリックされた場合
                        if event.button == 1:  # 左クリック: 位置移動
                            self.dragging_robot = i
                            robot.set_target_velocity(
                                0, 0, 0
                            )  # ドラッグ中は速度をゼロに
                        elif event.button == 3:  # 右クリック: 回転
                            self.rotating_robot = i
                            robot.set_target_velocity(
                                0, 0, 0
                            )  # ドラッグ中は速度をゼロに
                        break

                # ロボットでなければボールの近くかチェック（左クリックのみ）
                if (
                    self.dragging_robot is None
                    and self.rotating_robot is None
                    and event.button == 1
                ):
                    distance = ball.distance_to(mouse_x, mouse_y)
                    if distance < 20:  # ボールの近くでクリックされた場合
                        self.dragging_ball = True
                        ball.stop()  # ドラッグ中は速度をゼロに

            elif event.type == pygame.MOUSEBUTTONUP:
                # マウスボタンを離したらドラッグ・回転終了
                self.dragging_ball = False
                self.dragging_robot = None
                self.rotating_robot = None

            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                if self.dragging_ball:
                    # ドラッグ中はボールの位置をマウスに追従
                    ball.set_position(mouse_x, mouse_y)
                elif self.dragging_robot is not None:
                    # ドラッグ中はロボットの位置をマウスに追従
                    all_robots = robots + enemy_robots
                    all_robots[self.dragging_robot].x = mouse_x
                    all_robots[self.dragging_robot].y = mouse_y
                elif self.rotating_robot is not None:
                    # 右クリックドラッグ中はロボットの向きを変更
                    all_robots = robots + enemy_robots
                    robot = all_robots[self.rotating_robot]
                    # マウス位置への角度を計算（ロボットの向きとして設定）
                    # atan2は右向き(+x)を0として反時計回りに正
                    # ロボットの描画では-angleが使われるため、符号を反転
                    angle_to_mouse = math.atan2(mouse_y - robot.y, mouse_x - robot.x)
                    robot.angle = -angle_to_mouse

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Rキーが押されたらボールを初期位置に戻す
                    ball.reset()
                elif event.key == pygame.K_g:
                    # Gキーが押されたらロボットの位置と速度をリセット
                    for robot in robots:
                        robot.reset_position_and_velocity()
                    for robot in enemy_robots:
                        robot.reset_position_and_velocity()
                elif event.key == pygame.K_f:
                    # Fキーが押されたらボールをx,y方向に100m先にワープ
                    # シミュレーターでは1ピクセル=1cm、100m = 10000cm
                    ball.set_position(ball.x + 10000, ball.y + 10000)
                    ball.stop()  # 速度をゼロに

        return (
            self.dragging_ball
            or self.dragging_robot is not None
            or self.rotating_robot is not None
        )

import math

import pygame
import setting
from robot import Robot


class EnemyRobot(Robot):
    """敵ロボットクラス。Robotクラスを継承し、赤色で描画される"""

    # TODO: ロボットを検知する時、味方が敵、敵が味方に見えている。継承元のRobotクラスのメソッドから部分的に返る必要あり

    def reset_position_and_velocity(self):
        """敵ロボットの位置と速度を初期値にリセット"""
        # ロボット番号に応じた初期位置を設定（main.pyの初期配置と同じ）
        if self.robot_number == 1:
            self.x = 3 * setting.WIDTH // 4
            self.y = setting.HEIGHT // 3
            self.angle = math.pi
        elif self.robot_number == 2:
            self.x = 3 * setting.WIDTH // 4
            self.y = setting.HEIGHT // 2
            self.angle = math.pi
        elif self.robot_number == 3:
            self.x = 3 * setting.WIDTH // 4
            self.y = 2 * setting.HEIGHT // 3
            self.angle = math.pi
        else:
            # デフォルト位置
            self.x = 3 * setting.WIDTH // 4
            self.y = setting.HEIGHT // 2
            self.angle = math.pi

        # 速度を全てゼロにリセット
        self.vx = 0.0
        self.vy = 0.0
        self.angular_velocity = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_angular_velocity = 0.0
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_angle = self.angle
        self.interpolation_start_time = None

    def draw(self, screen):
        # フォントを初期化
        if pygame.get_init() and self.font is None:
            self.font = pygame.font.Font(None, 24)

        """敵ロボットを赤色で描画する"""
        # 親クラスのdrawメソッドの一部をコピーして、色を変更
        import math

        # 四角形の頂点を計算（中心が原点の場合）
        half_size = self.size // 2
        corners = [
            (-half_size, -half_size),
            (half_size, -half_size),
            (half_size, half_size),
            (-half_size, half_size),
        ]

        # 回転行列を適用して実際の位置に変換
        rotated_corners = []
        for x, y in corners:
            # 回転
            rot_x = x * math.cos(self.angle) - y * math.sin(self.angle)
            rot_y = x * math.sin(self.angle) + y * math.cos(self.angle)
            # 平行移動
            rotated_corners.append((self.x + rot_x, self.y + rot_y))

        # 回転した四角形を赤色で描画
        ENEMY_COLOR = (255, 0, 0)  # 赤色
        pygame.draw.polygon(screen, ENEMY_COLOR, rotated_corners)
        pygame.draw.polygon(screen, (255, 255, 255), rotated_corners, 2)  # 白い枠線

        # ロボットの向きを示す線を描画
        direction_length = self.size // 2 + 10
        end_x = self.x + direction_length * math.cos(self.angle)
        end_y = self.y + direction_length * math.sin(self.angle)
        pygame.draw.line(
            screen, (255, 255, 255), (self.x, self.y), (int(end_x), int(end_y)), 2
        )

        # ロボット番号を中心に描画（敵は"E"プレフィックス付き）
        text = self.font.render(f"E{self.robot_number}", True, (255, 255, 255))
        text_rect = text.get_rect(center=(int(self.x), int(self.y) + 15))
        screen.blit(text, text_rect)

import math
import time
import typing

import setting
import utils
from field_calc import calc_robot_and_object_relative_position, normalize_angle_rad
from numba import njit


@njit(cache=True)
def jit_get_vision_distance(
    fov_pitch_angle: float, neck_pitch: float, camera_height: float
) -> typing.Tuple[float, float]:
    """
    ロボットの首の角度に応じて物体を検出可能な距離を計算
    これは水平からの角度をピッチとする。
    この関数の実装より、sim内での角度は全て水平からの角度として扱うこと。
    """
    up_pitch_angle: float = neck_pitch - fov_pitch_angle / 2.0
    if up_pitch_angle < 0:
        up_pitch_angle = 0.0
    down_pitch_angle: float = neck_pitch + fov_pitch_angle / 2.0
    print(f"up_pitch_angle: {up_pitch_angle}, down_pitch_angle: {down_pitch_angle}")
    max_distance = min(camera_height * math.tan(math.pi / 2 - up_pitch_angle), 1500)
    minimum_distance = max(
        camera_height * math.tan(math.pi / 2 - down_pitch_angle), 40
    )  # 根元40cm以内は見えないとする
    vision_distance = (minimum_distance, max_distance)

    return vision_distance


class Robot:
    def __init__(
        self,
        robot_number: int,
        x=None,
        y=None,
        angle=None,
        complete_observation: bool = False,
    ):
        self.complete_observation = complete_observation
        self.robot_number = robot_number
        self.x = x if x is not None else setting.WIDTH // 4
        self.y = y if y is not None else setting.HEIGHT // 2
        self.vx = 0.0
        self.vy = 0.0
        if angle is not None:
            self.angle = angle
        else:
            self.angle = 0  # ラジアン単位
        self.angular_velocity = 0.0  # 角速度（ラジアン/秒）
        self.size = setting.ROBOT_SIZE
        # これはodom用だが、なんかかなり長いtermでの相対位置らしいので今の所は更新しない実装にする。あとで変えるかも
        self.prev_x = self.x  # 前回の位置(x)
        self.prev_y = self.y  # 前回の位置(y)
        self.prev_angle = self.angle  # 前回の角速度(ラジアン)

        # 首のyaw角度（ロボット座標系、-π/2 ~ π/2）
        self.neck_yaw = 0.0
        self.target_neck_yaw = 0.0
        self.neck_yaw_velocity = 0.0  # 首の角速度
        self.max_neck_yaw = 1.01229  # 最大首振り角度（58度）
        self.neck_yaw_speed = math.pi * 3  # 首の回転速度（ラジアン/秒）

        # 首のyaw角度（ロボット座標系、-π/2 ~ π/2）
        self.neck_pitch = 0.0
        self.target_neck_pitch = 0.0
        self.neck_pitch_velocity = 0.0  # 首の角速度
        self.max_neck_pitch = math.pi / 4.0  # 最大首振り角度（45度）
        self.neck_pitch_speed = math.pi * 3  # 首の回転速度（ラジアン/秒）

        # 視野角の設定
        self.fov_angle = 1.57  # 視野角 90度
        self.fov_range = 800  # 視野の距離（ピクセル）
        self.fov_pitch_angle = 1.127  # 視野の上下角度 65度
        self.camera_height = 102.0  # カメラの高さ(cm)（ピクセル）

        self.font = None

        # 速度補間用の変数
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_angular_velocity = 0.0
        self.interpolation_start_time = None
        self.interpolation_duration_x = 1.6  # 1.6秒
        self.interpolation_duration_y = 2.0  # 2.0秒
        self.interpolation_duration_theta = 1.9  # 1.9秒
        self.start_vx = 0.0
        self.start_vy = 0.0
        self.start_angular_velocity = 0.0
        self.cnt = 0

        # キャッシュ変数
        self._cached_vision_distance: typing.Optional[typing.Tuple[float, float]] = None
        self._cached_neck_pitch: typing.Optional[float] = None

    def set_neck_yaw(self, target_neck_yaw):
        """首のyaw角度を設定"""
        self.target_neck_yaw = utils.clamp(
            target_neck_yaw, -self.max_neck_yaw, self.max_neck_yaw
        )

    def set_neck_pitch(self, target_neck_pitch):
        """首のpitch角度を設定"""
        self.target_neck_pitch = utils.clamp(
            target_neck_pitch, 0.0, self.max_neck_pitch
        )

    def set_target_velocity(self, target_vx, target_vy, target_angular_velocity=None):
        """目標速度を設定し、速度補間を開始する"""
        self.target_vx = utils.clamp(
            target_vx, setting.ROBOT_MAX_SPEED_X_MINUS, setting.ROBOT_MAX_SPEED_X_PLUS
        )
        self.target_vy = utils.clamp(
            target_vy, -setting.ROBOT_MAX_SPEED_Y, setting.ROBOT_MAX_SPEED_Y
        )
        if target_angular_velocity is not None:
            self.target_angular_velocity = utils.clamp(
                -target_angular_velocity,
                -setting.ROBOT_MAX_SPEED_PI,
                setting.ROBOT_MAX_SPEED_PI,
            )
            self.start_angular_velocity = self.angular_velocity
        self.start_vx = self.vx
        self.start_vy = self.vy
        self.interpolation_start_time = time.time()

    def reset_position_and_velocity(self):
        """ロボットの位置と速度を初期値にリセット"""
        # ロボット番号に応じた初期位置を設定
        if self.robot_number == 1:
            self.x = 100
            self.y = setting.HEIGHT
            self.angle = math.pi / 2
        elif self.robot_number == 2:
            self.x = 300
            self.y = setting.HEIGHT
            self.angle = math.pi / 2
        elif self.robot_number == 3:
            self.x = 500
            self.y = setting.HEIGHT
            self.angle = math.pi / 2
        else:
            # デフォルト位置
            self.x = setting.WIDTH // 4
            self.y = setting.HEIGHT // 2
            self.angle = 0

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
        self.neck_yaw = 0.0
        self.target_neck_yaw = 0.0
        self.neck_pitch = 0.0
        self.target_neck_pitch = 0.0

    def update(self, disable_print=False):
        # 速度補間を適用
        if self.interpolation_start_time is not None:
            elapsed_time = time.time() - self.interpolation_start_time
            if elapsed_time < self.interpolation_duration_x:
                # 補間の進行度を計算（0から1）
                progress_x = utils.clamp(
                    elapsed_time / self.interpolation_duration_x, 0.0, 1.0
                )
                # 線形補間を使用して現在の速度を計算
                self.vx = self.start_vx + (self.target_vx - self.start_vx) * progress_x
            if elapsed_time < self.interpolation_duration_y:
                # 補間の進行度を計算（0から1）
                progress_y = utils.clamp(
                    elapsed_time / self.interpolation_duration_y, 0.0, 1.0
                )
                self.vy = self.start_vy + (self.target_vy - self.start_vy) * progress_y
            if elapsed_time < self.interpolation_duration_theta:
                # 補間の進行度を計算（0から1）
                progress_theta = utils.clamp(
                    elapsed_time / self.interpolation_duration_theta, 0.0, 1.0
                )
                self.angular_velocity = (
                    self.start_angular_velocity
                    + (self.target_angular_velocity - self.start_angular_velocity)
                    * progress_theta
                )
            else:
                # 補間完了
                self.vx = self.target_vx
                self.vy = self.target_vy
                self.angular_velocity = self.target_angular_velocity
                self.interpolation_start_time = None
        else:
            # 補間が完了している場合は、目標速度を維持
            self.vx = self.target_vx
            self.vy = self.target_vy
            self.angular_velocity = self.target_angular_velocity

        self.angle -= (
            self.angular_velocity / setting.SIM_FPS
        )  # 角度を更新（時計回りが正）

        # 首のyaw角度を更新
        if self.neck_yaw != self.target_neck_yaw:
            # 目標角度との差分を計算
            diff = self.target_neck_yaw - self.neck_yaw
            # 1フレームあたりの最大回転量
            max_rotation = self.neck_yaw_speed / setting.SIM_FPS

            if abs(diff) <= max_rotation:
                # 目標角度に到達
                self.neck_yaw = self.target_neck_yaw
            else:
                # 徐々に回転
                if diff > 0:
                    self.neck_yaw += max_rotation
                else:
                    self.neck_yaw -= max_rotation

        # 首のpitch角度を更新
        if self.neck_pitch != self.target_neck_pitch:
            # 目標角度との差分を計算
            diff = self.target_neck_pitch - self.neck_pitch
            # 1フレームあたりの最大回転量
            max_rotation = self.neck_pitch_speed / setting.SIM_FPS

            if abs(diff) <= max_rotation:
                # 目標角度に到達
                self.neck_pitch = self.target_neck_pitch
            else:
                # 徐々に回転
                if diff > 0:
                    self.neck_pitch += max_rotation
                else:
                    self.neck_pitch -= max_rotation

        # 速度を回転（角度の符号が反転したので回転行列も調整）
        cos_angle = math.cos(-self.angle)
        sin_angle = math.sin(-self.angle)
        global_vx = self.vx * cos_angle - self.vy * sin_angle
        global_vy = self.vx * sin_angle + self.vy * cos_angle

        # 位置更新（グローバル座標系で）
        self.x += global_vx / setting.SIM_FPS * 100.0
        self.y += global_vy / setting.SIM_FPS * 100.0

        self.cnt += 1
        if self.cnt % 120 == 0:
            if not disable_print:
                print(
                    f"Robot Position: ({self.x:.2f}, {self.y:.2f}), Velocity: ({self.vx:.2f}, {self.vy:.2f}), Ang vel: {self.angular_velocity:.2f} rad"
                )

    def get_odom(self):
        """ロボットの現在の位置と角度を返す"""
        diff_x, diff_y = (self.x - self.prev_x, self.y - self.prev_y)
        return (
            diff_x,
            diff_y,
            normalize_angle_rad(self.angle - self.prev_angle),
        )

    def get_vision_direction(self):
        """ロボットの視線方向（グローバル座標系）を取得"""
        # ロボットの向き + 首の角度
        vision_angle = -self.angle + self.neck_yaw
        return vision_angle

    def get_vision_distance(self) -> typing.Tuple[float, float]:
        """
        ロボットの首の角度に応じて物体を検出可能な距離を計算
        """
        if (
            self._cached_neck_pitch != self.neck_pitch
            or self._cached_vision_distance is None
        ):
            self._cached_vision_distance = jit_get_vision_distance(
                fov_pitch_angle=self.fov_pitch_angle,
                neck_pitch=self.neck_pitch,
                camera_height=self.camera_height,
            )
            self._cached_neck_pitch = self.neck_pitch

        return self._cached_vision_distance

    def is_in_view(self, target_x, target_y):
        """指定された位置がロボットの視野内にあるかチェック"""
        # 完全観測の場合は必ずTrueを返す
        if self.complete_observation:
            return True
        # ロボットから対象への相対位置
        dx = target_x - self.x
        dy = target_y - self.y
        distance_squared = dx * dx + dy * dy  # sqrt避ける

        # 視野範囲チェック（先に距離で絞る）
        min_distance, max_distance = self.get_vision_distance()
        if (
            distance_squared > max_distance * max_distance
            or distance_squared < min_distance * min_distance
        ):
            return False

        # 対象への角度を計算
        target_angle = math.atan2(dy, dx)

        # 視線方向を取得
        vision_angle = self.get_vision_direction()

        # 角度差を計算（-π ~ π の範囲に正規化）
        angle_diff = target_angle - vision_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # 視野角の半分以内にあるかチェック
        return abs(angle_diff) <= self.fov_angle / 2

    def draw(self, screen):
        import pygame

        # フォントを初期化
        if pygame.get_init() and self.font is None:
            self.font = pygame.font.Font(None, 24)

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
            # 回転（角度の符号が反転したので回転行列も調整）
            rot_x = x * math.cos(-self.angle) - y * math.sin(-self.angle)
            rot_y = x * math.sin(-self.angle) + y * math.cos(-self.angle)
            # 平行移動（整数に変換）
            rotated_corners.append((int(self.x + rot_x), int(self.y + rot_y)))

        # 回転した四角形を描画
        pygame.draw.polygon(screen, setting.ROBOT_COLOR, rotated_corners)
        pygame.draw.polygon(screen, setting.LINE_COLOR, rotated_corners, 2)

        # ロボットの向きを示す線を描画
        direction_length = self.size // 2 + 10
        end_x = self.x + direction_length * math.cos(-self.angle)
        end_y = self.y + direction_length * math.sin(-self.angle)
        pygame.draw.line(
            screen,
            setting.LINE_COLOR,
            (int(self.x), int(self.y)),
            (int(end_x), int(end_y)),
            2,
        )

        # ロボット番号を中心に描画
        text = self.font.render(str(self.robot_number), True, (255, 255, 255))
        text_rect = text.get_rect(center=(int(self.x), int(self.y) + 15))
        screen.blit(text, text_rect)

        # 視野角を描画
        self.draw_vision_cone(screen)

    def draw_vision_cone(self, screen):
        import pygame

        """視野角を扇形で描画"""
        # 視線方向を取得
        vision_angle = self.get_vision_direction()

        # 扇形の頂点（ロボットの位置）
        center = (int(self.x), int(self.y))

        fov_distance_min, fov_distance_max = self.get_vision_distance()

        # リング状の扇形を構成する点のリストを作成
        points = []

        # 外側の弧を構成する点を追加（視野角の範囲）
        num_points = 20  # 弧の滑らかさ
        for i in range(num_points + 1):
            # 視野角の範囲内で角度を計算
            angle = (
                vision_angle - self.fov_angle / 2 + (self.fov_angle * i / num_points)
            )
            # 点の座標を計算（外側の弧）
            x = self.x + fov_distance_max * math.cos(angle)
            y = self.y + fov_distance_max * math.sin(angle)
            points.append((int(x), int(y)))

        # 内側の弧を構成する点を追加（逆順で）
        for i in range(num_points, -1, -1):
            # 視野角の範囲内で角度を計算
            angle = (
                vision_angle - self.fov_angle / 2 + (self.fov_angle * i / num_points)
            )
            # 点の座標を計算（内側の弧）
            x = self.x + fov_distance_min * math.cos(angle)
            y = self.y + fov_distance_min * math.sin(angle)
            points.append((int(x), int(y)))

        # 半透明のリング状扇形を描画
        # Pygameでは直接透明度を扱えないので、別のサーフェスに描画してからブレンド
        s = pygame.Surface((setting.WIDTH, setting.HEIGHT), pygame.SRCALPHA)
        # 視野角を半透明の黄色で描画
        pygame.draw.polygon(s, (255, 255, 0, 40), points)
        screen.blit(s, (0, 0))

        # 視野角の境界線を描画
        # 左側の境界線（内側から外側へ）
        left_angle = vision_angle - self.fov_angle / 2
        left_x_min = self.x + fov_distance_min * math.cos(left_angle)
        left_y_min = self.y + fov_distance_min * math.sin(left_angle)
        left_x_max = self.x + fov_distance_max * math.cos(left_angle)
        left_y_max = self.y + fov_distance_max * math.sin(left_angle)
        pygame.draw.line(
            screen,
            (255, 255, 0),
            (int(left_x_min), int(left_y_min)),
            (int(left_x_max), int(left_y_max)),
            1,
        )

        # 右側の境界線（内側から外側へ）
        right_angle = vision_angle + self.fov_angle / 2
        right_x_min = self.x + fov_distance_min * math.cos(right_angle)
        right_y_min = self.y + fov_distance_min * math.sin(right_angle)
        right_x_max = self.x + fov_distance_max * math.cos(right_angle)
        right_y_max = self.y + fov_distance_max * math.sin(right_angle)
        pygame.draw.line(
            screen,
            (255, 255, 0),
            (int(right_x_min), int(right_y_min)),
            (int(right_x_max), int(right_y_max)),
            1,
        )

        # 視線方向の中心線（赤色で強調、内側から外側へ）
        center_x_min = self.x + fov_distance_min * math.cos(vision_angle)
        center_y_min = self.y + fov_distance_min * math.sin(vision_angle)
        center_x_max = self.x + fov_distance_max * math.cos(vision_angle)
        center_y_max = self.y + fov_distance_max * math.sin(vision_angle)
        pygame.draw.line(
            screen,
            (255, 0, 0),
            (int(center_x_min), int(center_y_min)),
            (int(center_x_max), int(center_y_max)),
            2,
        )

    def get_detected_field_landmarks(
        self, landmark_list: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        引数として渡したロボットが引数のランドマークを観測しているならばそれの相対座標のリストを返す
        """
        detected_landmarks = []
        for landmark in landmark_list:
            if self.is_in_view(landmark[0], landmark[1]):
                x, y = calc_robot_and_object_relative_position(
                    self.x,
                    self.y,
                    self.angle,
                    landmark[0],
                    landmark[1],
                    scale_x=setting.DETECTION_DISTANCE_SCALE_X,
                    scale_y=setting.DETECTION_DISTANCE_SCALE_Y,
                )
                detected_landmarks.append((x, y))
        return detected_landmarks

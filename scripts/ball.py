import setting
from numpy import mean, percentile, random, sqrt, var


class Ball:
    def __init__(self, x=None, y=None):
        self.x = x if x is not None else setting.WIDTH // 2
        self.y = y if y is not None else setting.HEIGHT // 2
        self.vx = 0
        self.vy = 0
        self.radius = 11
        self.damping_factor = 0.99

        # 飛距離制御用の変数
        self.target_distance = 0
        self.initial_position = (self.x, self.y)
        self.distance_travelled = 0
        self.deceleration_active = False

        # ゴール判定用
        self.ball_in_goal = False

    def update(self):
        if self.deceleration_active and self.target_distance > 0:
            # 現在までの移動距離を計算
            dx = self.x - self.initial_position[0]
            dy = self.y - self.initial_position[1]
            self.distance_travelled = sqrt(dx**2 + dy**2)

            # 目標距離に近づいたら減速を強める
            if self.distance_travelled < self.target_distance:
                # 残り距離の割合に基づいて減速
                remaining_ratio = 1 - (self.distance_travelled / self.target_distance)

                # 減速係数を計算（目標に近づくほど強い減速）
                dynamic_damping = 0.85 + (0.14 * remaining_ratio)  # 0.85 ~ 0.99
                self.vx *= dynamic_damping
                self.vy *= dynamic_damping
            else:
                # 目標距離に到達したら停止
                self.vx = 0
                self.vy = 0
                self.deceleration_active = False
        else:
            # 通常の減衰を適用
            self.vx *= self.damping_factor
            self.vy *= self.damping_factor

        # 位置を更新
        self.x += self.vx
        self.y += self.vy

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def set_velocity(self, vx, vy):
        self.vx = vx
        self.vy = vy

    def reset(self):
        self.set_position(setting.WIDTH // 2, setting.HEIGHT // 2)
        self.stop()

    # direction_xとdirection_yにはロボットの速度を入れる
    def kick_with_distance(self, direction_x, direction_y):
        """指定した方向と飛距離でボールをキックする"""
        # 方向ベクトルを正規化
        mag_x, mag_y = ball_distance_kicked_by_robot(direction_x, direction_y)
        magnitude = sqrt(mag_x**2 + mag_y**2)
        if magnitude > 0:
            mag_x /= magnitude
            mag_y /= magnitude

        # 飛距離制御を有効化
        self.target_distance = magnitude
        self.initial_position = (self.x, self.y)
        self.distance_travelled = 0
        self.deceleration_active = True

        # 初期速度を計算（飛距離に基づいて）
        # 簡単な物理モデル：v = sqrt(2 * deceleration * distance)
        # ここでは単純化して、距離に比例した初期速度を設定
        initial_speed = magnitude * 0.1  # 調整可能な係数
        self.vx = mag_x * initial_speed
        self.vy = mag_y * initial_speed

    def stop(self):
        self.vx = 0
        self.vy = 0
        self.deceleration_active = False
        self.target_distance = 0

    def draw(self, screen):
        from pygame import draw

        draw.circle(screen, setting.BALL_COLOR, (int(self.x), int(self.y)), self.radius)

    def distance_to(self, x, y):
        dx = self.x - x
        dy = self.y - y
        return sqrt(dx * dx + dy * dy)

    def check_ball_in_goal(self):
        """ボールがゴールに入ったかを判定する"""
        # 左ゴール（x=0側）の判定
        if -100 <= self.x <= 0:
            if setting.GOALPOST_LEFT_UP[1] <= self.y <= setting.GOALPOST_LEFT_DOWN[1]:
                if self.ball_in_goal:
                    return setting.GoalType.NONE
                self.ball_in_goal = True
                return setting.GoalType.LEFT  # 味方ゴール

        # 右ゴール（x=WIDTH側）の判定
        if setting.WIDTH + 100 >= self.x >= setting.WIDTH:
            if setting.GOALPOST_RIGHT_UP[1] <= self.y <= setting.GOALPOST_RIGHT_DOWN[1]:
                if self.ball_in_goal:
                    return setting.GoalType.NONE
                self.ball_in_goal = True
                return setting.GoalType.RIGHT  # 敵ゴール

        self.ball_in_goal = False
        return setting.GoalType.NONE


class BallExperimentData:
    def __init__(self):
        self.data_x: list[float] = []
        self.data_y: list[float] = []
        self.mean_x: float = 0.0
        self.mean_y: float = 0.0
        self.var_x: float = 0.0
        self.var_y: float = 0.0


class BallDistribution:
    def __init__(self):
        # x速度の時のボールの飛距離データ
        # ペナルティ -> センター = 7 - 2.1 = 4.9m
        self.x_vel_data = BallExperimentData()
        self.x_vel_data.data_x = [
            4.9 - 0.35,
            2.1,
            2.5,
            2.8,
            2.4,
            4.9 - 0.85,
            2.3,
            0.45,
            4.9 - 0.25,
            1.85,
            2.25,
            2.67,
            0.16,
            1.65,
            3.9,
            2.65,
            2.1,
            0.9,
            3.9,
        ]
        self.x_vel_data.data_y = [
            1.3,
            -0.25,
            -0.8,
            -0.45,
            0.1,
            0.15,
            -0.4,
            0.25,
            0.15,
            1.3,
            -2.3,
            1.22,
            -0.2,
            -1.95,
            -0.64,
            -0.15,
            2.15,
            -0.52,
            0.38,
        ]
        self.x_vel_data.mean_x, self.x_vel_data.var_x = self.bootstrap_estimate(
            self.x_vel_data.data_x
        )
        self.x_vel_data.mean_y, self.x_vel_data.var_y = self.bootstrap_estimate(
            self.x_vel_data.data_y
        )
        # y速度の時のボールの飛距離データ
        self.y_vel_data = BallExperimentData()
        self.y_vel_data.data_y = [
            0.8,
            0.5,
            0.85,
            0.55,
            0.8,
            0.2,
            0.4,
            0.2,
            0.17,
            0.1,
            0.17,
            0.14,
            0.1,
            0.2,
        ]
        self.y_vel_data.data_x = [
            0.25,
            0.25,
            0.25,
            -0.1,
            -0.1,
            0.35,
            0.5,
            0.75,
            0.15,
            0.67,
            0.75,
            0.22,
            0.54,
            0.7,
        ]
        self.y_vel_data.mean_y, self.y_vel_data.var_y = self.bootstrap_estimate(
            self.y_vel_data.data_y
        )
        self.y_vel_data.mean_x, self.y_vel_data.var_x = self.bootstrap_estimate(
            self.y_vel_data.data_x
        )

    def bootstrap_estimate(self, data, n_bootstrap=10000):
        n = len(data)
        bootstrap_means = []
        bootstrap_vars = []

        for _ in range(n_bootstrap):
            # 復元抽出
            sample = random.choice(data, size=n, replace=True)
            bootstrap_means.append(mean(sample))
            bootstrap_vars.append(var(sample, ddof=1))

        # 信頼区間の計算（95%）
        mean_ci = percentile(bootstrap_means, [2.5, 97.5])
        var_ci = percentile(bootstrap_vars, [2.5, 97.5])
        return mean(data), var(data, ddof=1)

    def get_ball_distance_by_x_velo(self, robot_vel_x):
        persentile_x = robot_vel_x / setting.ROBOT_MAX_SPEED_X_PLUS
        mean_x, var_x = self.x_vel_data.mean_x, self.x_vel_data.var_x
        mean_y, var_y = self.x_vel_data.mean_y, self.x_vel_data.var_y
        x = random.normal(mean_x, sqrt(var_x))
        y = random.normal(mean_y, sqrt(var_y))
        return x * persentile_x, y * persentile_x

    def get_ball_distance_by_y_velo(self, robot_vel_y):
        persentile_y = robot_vel_y / setting.ROBOT_MAX_SPEED_Y
        mean_x, var_x = self.y_vel_data.mean_x, self.y_vel_data.var_x
        mean_y, var_y = self.y_vel_data.mean_y, self.y_vel_data.var_y
        x = random.normal(mean_x, sqrt(var_x))
        y = random.normal(mean_y, sqrt(var_y))
        return x * persentile_y, y * persentile_y

    def _ball_distance_kicked_by_robot(self, robot_vx, robot_vy):
        result_x, result_y = 0, 0
        x, y = self.get_ball_distance_by_x_velo(robot_vx)
        result_x += x
        result_y += y
        x, y = self.get_ball_distance_by_y_velo(robot_vy)
        result_x += x
        result_y += y

        return (
            result_x * 100.0,
            result_y * 100.0,
        )  # シミュレーターでは1ピクセル=1cmなので、100倍してmに変換


ball_distribution = BallDistribution()


def ball_distance_kicked_by_robot(robot_vx, robot_vy):
    global ball_distribution
    return ball_distribution._ball_distance_kicked_by_robot(robot_vx, robot_vy)

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import setting
from numpy import fabs, random, sqrt

# 歩行中の誤差の分散(最大遠の時の誤差)
BALL_AVERAGE_WALKING_X: int = 13 * 2
BALL_AVERAGE_WALKING_Y: int = 9 * 2


# 立っている時の誤差の分散(最大遠の時の誤差)
BALL_AVERAGE_STANDING_X: int = 10
BALL_AVERAGE_STANDING_Y: int = 5

# 自己位置推定の誤差の分散
SELFPOS_COVARIANCE_X: int = 30
SELFPOS_COVARIANCE_Y: int = 30
SELFPOS_ERROR_MAX_CONTINUOUS_TIME: float = (
    3.0 * setting.SIM_FPS
)  # 誤差が継続する最大時間(秒)


class ObservationRandomizerBase(ABC):
    @abstractmethod
    def reseed(self) -> None:
        pass

    @abstractmethod
    def get_seed(self) -> int:
        pass

    @abstractmethod
    def get_ball_error_xy(
        self, robot_vx: float, robot_vy: float, ball_rel_x: float, ball_rel_y: float
    ) -> Tuple[float, float]:
        pass

    @abstractmethod
    def get_selfpos_error_xy(self) -> Tuple[float, float]:
        pass


class SelfposError:
    def __init__(self, rng):
        self.rng = rng
        self.error_x = self.rng.normal(loc=0.0, scale=SELFPOS_COVARIANCE_X)
        self.error_y = self.rng.normal(loc=0.0, scale=SELFPOS_COVARIANCE_Y)
        self.continous_time = self.rng.uniform(0.0, SELFPOS_ERROR_MAX_CONTINUOUS_TIME)
        self.current_time = 0

    def get_current_error(self):
        if self.continous_time < self.current_time:
            self.error_x = self.rng.normal(loc=0.0, scale=SELFPOS_COVARIANCE_X)
            self.error_y = self.rng.normal(loc=0.0, scale=SELFPOS_COVARIANCE_Y)
            self.continous_time = self.rng.uniform(
                0.0, SELFPOS_ERROR_MAX_CONTINUOUS_TIME
            )
            self.current_time = 0
        self.current_time += 1
        return self.error_x, self.error_y


class Randomize(ObservationRandomizerBase):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        if seed is None:
            self.seed = (
                int(time.time() * 1000) % 2**32
            )  # Use current time in milliseconds as seed
        else:
            self.seed = seed
        self.rng = random.default_rng(seed=self.seed)
        self.selfpos_err = SelfposError(self.rng)

    def reseed(self):
        self.seed = int(time.time() * 1000) % 2**32
        self.rng = random.default_rng(seed=self.seed)

    def get_seed(self):
        return self.seed

    def _impl_calc_ball_error_scale(
        self, ball_rel_x: float, ball_rel_y: float
    ) -> float:
        distance = min(
            900.0, sqrt(ball_rel_x**2 + ball_rel_y**2)
        )  # 900mm以上は誤差が変わらないとする
        return distance / 900.0  # 大体9m位が見える最大の距離？

    def get_ball_error_xy(
        self, robot_vx: float, robot_vy: float, ball_rel_x: float, ball_rel_y: float
    ) -> Tuple[float, float]:
        if fabs(robot_vx) < 0.01 and fabs(robot_vy) < 0.01:
            std_dev_x = BALL_AVERAGE_STANDING_X
            std_dev_y = BALL_AVERAGE_STANDING_Y
        else:
            std_dev_x = BALL_AVERAGE_WALKING_X
            std_dev_y = BALL_AVERAGE_WALKING_Y

        # ボールとの距離に応じて誤差の大きさを変更する
        std_dev_x = std_dev_x * self._impl_calc_ball_error_scale(ball_rel_x, ball_rel_y)
        std_dev_y = std_dev_y * self._impl_calc_ball_error_scale(ball_rel_x, ball_rel_y)
        error_x = self.rng.normal(loc=0.0, scale=std_dev_x)
        error_y = self.rng.normal(loc=0.0, scale=std_dev_y)

        return error_x, error_y

    def get_selfpos_error_xy(self) -> Tuple[float, float]:
        return self.selfpos_err.get_current_error()


if __name__ == "__main__":
    rand = Randomize()
    robot_vx = 0.5
    robot_vy = 0.5
    error_x, error_y = rand.get_ball_error_xy(robot_vx, robot_vy, 6, 6)
    print(f"Ball error (X): {error_x}, Ball error (Y): {error_y}")
    robot_vx = 0.0
    robot_vy = 0.0
    error_x, error_y = rand.get_ball_error_xy(robot_vx, robot_vy, 1, 1)
    print(f"Ball error (X): {error_x}, Ball error (Y): {error_y}")

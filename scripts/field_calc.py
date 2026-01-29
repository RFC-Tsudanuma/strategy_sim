import numpy as np
import setting
from numba import njit


def convert_to_robot_coord(x, y):
    """
    ロボット内部で利用する座標系に変換する。
    フィールドの中心を(0,0)として、x軸は右方向、y軸は上方向とする。
    """
    return x - (setting.WIDTH / 2), (setting.HEIGHT / 2) - y


@njit(cache=True)
def rotation_matrix_2d(angle_rad):
    """2D回転行列を作成（rad）"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


@njit(cache=True)
def calc_robot_and_object_relative_position(
    robot_x, robot_y, robot_theta, object_x, object_y, scale_x=1.0, scale_y=1.0
):
    """
    Calculate the relative position of an object with respect to a robot.
    オブジェクトとロボットの両方とも、座標系が同じであること！
    ロボットの座標系は右手系なので、y軸はマイナスを掛けている。
    """
    rel_x = object_x - robot_x
    rel_y = object_y - robot_y
    coord = np.array([rel_x, rel_y]).astype(np.float64)
    rotation_mat = rotation_matrix_2d(robot_theta)
    result = rotation_mat @ coord
    return (result[0] * scale_x, -result[1] * scale_y)


@njit(cache=True)
def normalize_angle_rad(angle_rad):
    """角度を-rad.pi～rad.piの範囲に正規化する"""
    while angle_rad > np.pi:
        angle_rad -= 2.0 * np.pi
    while angle_rad < -np.pi:
        angle_rad += 2.0 * np.pi
    return angle_rad

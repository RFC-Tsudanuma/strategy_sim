from torch import Tensor, cos, sin, tensor


def robot_velocity_vec(x_vel: float, y_vel: float, selfpos_theta: float) -> Tensor:
    """
    ロボットの自己位置とオドメトリから速度ベクトルを計算する関数
    """
    # オドメトリの速度成分をロボット座標系に変換
    cos_theta = cos(tensor(selfpos_theta))
    sin_theta = sin(tensor(selfpos_theta))

    vx_robot = cos_theta * x_vel + sin_theta * y_vel
    vy_robot = -sin_theta * x_vel + cos_theta * y_vel

    return tensor([vx_robot, vy_robot])

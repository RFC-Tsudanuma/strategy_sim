import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../.."))

from numpy import ndarray
from torch import Tensor
from training_interface import (
    InputNeckCommand,
    InputWalkCommand,
)


def convert_action_tensor_to_input(
    action: Tensor,
) -> tuple[InputWalkCommand, InputNeckCommand]:
    # バッチ次元がある場合は最初の要素を取得
    if action.dim() > 1:
        action = action.squeeze(0)
    walk = InputWalkCommand()
    walk.x_velocity = action[0].item()
    walk.y_velocity = action[1].item()
    walk.theta_velocity = action[2].item()
    neck = InputNeckCommand()
    neck.neck_yaw_angle = action[3].item()
    neck.neck_pitch_angle = action[4].item()
    return walk, neck


def convert_tensor_to_np(action: Tensor) -> ndarray:
    if action.dim() > 1:
        action = action.squeeze(0)
    return action.cpu().numpy()

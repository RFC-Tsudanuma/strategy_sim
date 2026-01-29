import os
import sys

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../scripts")
)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../scripts"))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../.."))

import matplotlib.pyplot as plt
import numpy as np
from env_config import DRConfig
from numpy import set_printoptions

if __name__ == "__main__":
    dr = DRConfig(seed=42)
    set_printoptions(precision=4, suppress=True)

    robot_x_list = []
    robot_y_list = []
    robot_w_list = []
    ball_x_list = []
    ball_y_list = []

    for _ in range(1000):
        x, y, w = dr.init_robot(1)
        ball_x, ball_y = dr.init_ball()

        robot_x_list.append(x)
        robot_y_list.append(y)
        robot_w_list.append(w)
        ball_x_list.append(ball_x)
        ball_y_list.append(ball_y)

    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ロボットの位置分布
    axes[0, 0].scatter(robot_x_list, robot_y_list, alpha=0.5, s=10)
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")
    axes[0, 0].set_title("Robot Position Distribution")
    axes[0, 0].grid(True)
    axes[0, 0].axis("equal")

    # ロボットの角度分布
    axes[0, 1].hist(robot_w_list, bins=50, alpha=0.7)
    axes[0, 1].set_xlabel("Angle (rad)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Robot Angle Distribution")
    axes[0, 1].grid(True)

    # ボールの位置分布
    axes[1, 0].scatter(ball_x_list, ball_y_list, alpha=0.5, s=10, color="orange")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    axes[1, 0].set_title("Ball Position Distribution")
    axes[1, 0].grid(True)
    axes[1, 0].axis("equal")

    # 統計情報
    stats_text = f"""Robot Stats:
X: mean={np.mean(robot_x_list):.3f}, std={np.std(robot_x_list):.3f}
Y: mean={np.mean(robot_y_list):.3f}, std={np.std(robot_y_list):.3f}
W: mean={np.mean(robot_w_list):.3f}, std={np.std(robot_w_list):.3f}

Ball Stats:
X: mean={np.mean(ball_x_list):.3f}, std={np.std(ball_x_list):.3f}
Y: mean={np.mean(ball_y_list):.3f}, std={np.std(ball_y_list):.3f}"""

    axes[1, 1].text(
        0.1,
        0.5,
        stats_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Statistics")

    plt.tight_layout()
    plt.savefig("randomize_debug.png", dpi=150)
    print("Plot saved to randomize_debug.png")
    plt.show()

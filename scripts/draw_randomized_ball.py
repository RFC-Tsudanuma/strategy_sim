#!/usr/bin/env python3
import pygame


def draw_randomized_ball(
    screen, robot_number, randomized_x, randomized_y, ball_radius=11
):
    """
    ロボットから見たランダム化されたボール位置を描画する

    Args:
        screen: pygame screen object
        robot_number: ロボット番号 (1, 2, 3)
        randomized_x: エラーを含んだボールのX座標
        randomized_y: エラーを含んだボールのY座標
        ball_radius: ボールの半径（デフォルト11）
    """
    # 各ロボットごとの色設定
    colors = {
        1: (255, 100, 100),  # Robot 1: 赤
        2: (150, 255, 150),  # Robot 2: 緑
        3: (100, 100, 255),  # Robot 3: 青
    }

    # デフォルト色（想定外のロボット番号の場合）
    color = colors.get(robot_number, (200, 200, 200))

    # ランダム化されたボール位置に円の輪郭を描画
    pygame.draw.circle(
        screen,
        color,
        (int(randomized_x), int(randomized_y)),
        ball_radius,
        4,  # 線の太さ（塗りつぶさない輪郭線）
    )

import math


def check_robot_robot_collision(robot1, robot2):
    """2つのロボット間の衝突をチェックし、衝突している場合は移動を制限する"""
    # 衝突判定距離（両ロボットのサイズの半分の和）
    collision_distance = robot1.size // 2 + robot2.size // 2

    # 中心間の距離を計算
    dx = robot2.x - robot1.x
    dy = robot2.y - robot1.y
    distance = math.sqrt(dx * dx + dy * dy)

    # 衝突判定
    if distance < collision_distance:
        if distance > 0:  # ゼロ除算を防ぐ
            # 正規化された方向ベクトル
            nx = dx / distance
            ny = dy / distance

            # 各ロボットの速度の相手方向成分を計算
            v1_dot = robot1.vx * nx + robot1.vy * ny
            v2_dot = robot2.vx * (-nx) + robot2.vy * (-ny)

            # 相手に向かって移動している場合、その成分を除去
            if v1_dot > 0:
                robot1.vx -= v1_dot * nx
                robot1.vy -= v1_dot * ny

            if v2_dot > 0:
                robot2.vx -= v2_dot * (-nx)
                robot2.vy -= v2_dot * (-ny)

            # 重なりを解消するための押し出し
            overlap = collision_distance - distance
            push_x = nx * overlap * 0.5
            push_y = ny * overlap * 0.5

            robot1.x -= push_x
            robot1.y -= push_y
            robot2.x += push_x
            robot2.y += push_y

            return True

    return False


def check_robot_ball_collision(robot, ball):
    """ロボットとボールの衝突をチェックし、衝突時の処理を行う

    Args:
        robot: ロボットオブジェクト
        ball: ボールオブジェクト
        kick_distance: ボールが飛ぶ距離（デフォルト100ピクセル）
    """
    # 衝突判定距離
    collision_distance = ball.radius + robot.size // 2

    # 中心間の距離を計算
    dx = ball.x - robot.x
    dy = ball.y - robot.y
    distance = math.sqrt(dx * dx + dy * dy)

    # 衝突判定
    if distance < collision_distance:
        # 衝突時の処理
        # ボールを押し出して重なりを解消
        if distance > 0:  # ゼロ除算を防ぐ
            push_distance = collision_distance - distance
            push_x = (dx / distance) * push_distance
            push_y = (dy / distance) * push_distance
            ball.x += push_x
            ball.y += push_y

            # ボールを指定距離でキック
            # ロボットの速度ベクトルを基準にキック方向を決定
            if math.fabs(robot.vx) > 1e-6 or math.fabs(robot.vy) > 1e-6:
                # ロボットが動いている場合は速度方向にキック
                # ロボットのローカル座標系の速度をグローバル座標系に変換
                cos_angle = math.cos(-robot.angle)
                sin_angle = math.sin(-robot.angle)
                global_vx = robot.vx * cos_angle - robot.vy * sin_angle
                global_vy = robot.vx * sin_angle + robot.vy * cos_angle
                ball.kick_with_distance(global_vx, global_vy)
            else:
                # ロボットが静止している場合はロボットの所で停止
                ball.stop()

        return True

    return False

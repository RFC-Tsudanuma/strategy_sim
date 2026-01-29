import setting


def draw_field(screen):
    from pygame import draw

    # フィールドの線を描画
    screen.fill(setting.FIELD_COLOR)
    draw.rect(
        screen, setting.LINE_COLOR, (10, 10, setting.WIDTH - 20, setting.HEIGHT - 20), 3
    )
    draw.line(
        screen,
        setting.LINE_COLOR,
        (setting.WIDTH // 2, 10),
        (setting.WIDTH // 2, setting.HEIGHT - 10),
        3,
    )
    draw.circle(
        screen, setting.LINE_COLOR, (setting.WIDTH // 2, setting.HEIGHT // 2), 150, 3
    )

    # ペナルティエリアの設定
    penalty_width = setting.PENALTY_WIDTH
    penalty_height = setting.PENALTY_HEIGHT

    # 左側のペナルティエリア
    draw.rect(
        screen,
        setting.LINE_COLOR,
        (10, (setting.HEIGHT - penalty_height) // 2, penalty_width, penalty_height),
        3,
    )

    # 右側のペナルティエリア
    draw.rect(
        screen,
        setting.LINE_COLOR,
        (
            setting.WIDTH - 10 - penalty_width,
            (setting.HEIGHT - penalty_height) // 2,
            penalty_width,
            penalty_height,
        ),
        3,
    )

    # ゴールエリア（小さい方の四角）
    goal_area_width = setting.GOALAREA_WIDTH
    goal_area_height = setting.GOALAREA_HEIGHT

    # 左側のゴールエリア
    draw.rect(
        screen,
        setting.LINE_COLOR,
        (
            10,
            (setting.HEIGHT - goal_area_height) // 2,
            goal_area_width,
            goal_area_height,
        ),
        3,
    )

    # 右側のゴールエリア
    draw.rect(
        screen,
        setting.LINE_COLOR,
        (
            setting.WIDTH - 10 - goal_area_width,
            (setting.HEIGHT - goal_area_height) // 2,
            goal_area_width,
            goal_area_height,
        ),
        3,
    )

    # ペナルティスポット
    penalty_spot_distance = 210
    draw.circle(
        screen, setting.LINE_COLOR, (10 + penalty_spot_distance, setting.HEIGHT // 2), 5
    )
    draw.circle(
        screen,
        setting.LINE_COLOR,
        (setting.WIDTH - 10 - penalty_spot_distance, setting.HEIGHT // 2),
        5,
    )

    # ゴールの描画
    goal_height = 260
    draw.line(
        screen,
        setting.GOAL_COLOR,
        (0, setting.HEIGHT // 2 - goal_height // 2),
        (0, setting.HEIGHT // 2 + goal_height // 2),
        5,
    )
    draw.line(
        screen,
        setting.GOAL_COLOR,
        (setting.WIDTH, setting.HEIGHT // 2 - goal_height // 2),
        (setting.WIDTH, setting.HEIGHT // 2 + goal_height // 2),
        5,
    )
    draw.rect(
        screen,
        setting.GOAL_COLOR,
        (0, setting.HEIGHT // 2 - goal_height // 2, 10, 260),
        0,
    )
    draw.rect(
        screen,
        setting.GOAL_COLOR,
        (setting.WIDTH - 10, setting.HEIGHT // 2 - goal_height // 2, 10, 260),
        0,
    )

import setting
from training_interface import GameInfo


class GameState:
    """
    試合の状況を管理するためのクラス
    """

    def __init__(self):
        self.ally_score: int = 0
        self.enemy_score: int = 0
        self.time_remaining_s: float = 1200.0  # 試合時間の残り (秒)
        self.last_scored: setting.GoalType = setting.GoalType.NONE

    def reset(self):
        """試合の状況を初期化する"""
        self.ally_score = 0
        self.enemy_score = 0
        self.time_remaining_s = 1200.0

    def update_time(self, delta_time: float):
        """試合時間を更新する"""
        self.time_remaining_s = max(0.0, self.time_remaining_s - delta_time)

    def update_score(self, goal_type: setting.GoalType):
        """スコアを更新する"""
        if goal_type == setting.GoalType.LEFT:
            self.ally_score += 1
        elif goal_type == setting.GoalType.RIGHT:
            self.enemy_score += 1
        self.last_scored = goal_type  # 今回の得点チームを記録

    def state_string(self) -> str:
        """試合の状況(時間、スコア)を文字列で返す"""
        if self.time_remaining_s > 600:
            half_time = self.time_remaining_s - 600
            half = "First Half"
        else:
            half_time = self.time_remaining_s
            half = "Second Half"
        minutes = int(half_time) // 60
        seconds = int(half_time) % 60
        return f"Ally: {self.ally_score} - Enemy: {self.enemy_score}\n{half}: {minutes:02}:{seconds:02}"

    def draw_state(self, screen):
        """試合の状況(時間、スコア)を画面に描画する"""
        import pygame

        font = pygame.font.Font(None, 36)
        text = self.state_string()
        lines = text.split("\n")

        screen_width = screen.get_width()

        for i, line in enumerate(lines):
            text_surface = font.render(line, True, (255, 255, 255))
            text_width = text_surface.get_width()
            x_position = (screen_width - text_width) // 2 + 120
            y_position = 30 + i * 40
            screen.blit(text_surface, (x_position, y_position))

    def to_game_info(self) -> GameInfo:
        """試合の状況をGameInfo形式で返す"""
        return GameInfo(
            our_score=self.ally_score,
            opponent_score=self.enemy_score,
            time_remaining=self.time_remaining_s,
            scored_team=self.last_scored,
        )

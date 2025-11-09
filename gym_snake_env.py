import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game import SnakeGame, Direction, Point

class SnakeGymEnv(gym.Env):
    """
    Gymnasium wrapper around your SnakeGame.
    Observation: 11-dim vector (0/1) matching Agent.get_state
    Action: Discrete(3) corresponding to [straight, right, left] (same semantics as your agent)
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = SnakeGame()
        self.render_mode = render_mode

        # Observation is 11 binary features, but use float32 box for compat with baselines
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)

        # Actions: 3 discrete choices used by your agent ([1,0,0] etc.)
        self.action_space = spaces.Discrete(3)

    def _state_from_game(self):
        # replicate Agent.get_state logic but return float32 array
        head = self.game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            (dir_r and self.game.is_collision(point_r)) or
            (dir_l and self.game.is_collision(point_l)) or
            (dir_u and self.game.is_collision(point_u)) or
            (dir_d and self.game.is_collision(point_d)),

            (dir_u and self.game.is_collision(point_r)) or
            (dir_d and self.game.is_collision(point_l)) or
            (dir_l and self.game.is_collision(point_u)) or
            (dir_r and self.game.is_collision(point_d)),

            (dir_d and self.game.is_collision(point_r)) or
            (dir_u and self.game.is_collision(point_l)) or
            (dir_r and self.game.is_collision(point_u)) or
            (dir_l and self.game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            self.game.food.x < head.x,
            self.game.food.x > head.x,
            self.game.food.y < head.y,
            self.game.food.y > head.y
        ]
        return np.array(state, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # reset and return initial observation and info (gymnasium API)
        super().reset(seed=seed)
        self.game.reset()
        obs = self._state_from_game()
        return obs, {}

    def step(self, action):
        """
        action: integer 0/1/2 (Discrete)
        Convert to final_move list expected by game.play_step
        """
        if not isinstance(action, (int, np.integer)):
            action = int(action)

        final_move = [0, 0, 0]
        final_move[action] = 1

        # your game.play_step returns (game_over, score, reward)
        game_over, score, reward = self.game.play_step(final_move)
        obs = self._state_from_game()
        terminated = bool(game_over)
        truncated = False
        info = {"score": score}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        # If your SnakeGame uses pygame and has a render/display method, call it.
        # Otherwise do nothing or implement a matplotlib view.
        try:
            # try to call an existing render method, if present
            if hasattr(self.game, "render"):
                self.game.render()
        except Exception:
            pass

    def close(self):
        try:
            if hasattr(self.game, "close"):
                self.game.close()
        except Exception:
            pass
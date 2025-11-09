import os
import time
import signal
import atexit
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gym_snake_env import SnakeGymEnv

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# global model reference for safe saving from signal/atexit/callback
_model_for_saving = None

def _safe_save_model(base_path="models/snake_dqn_model"):
    global _model_for_saving
    if _model_for_saving is None:
        return
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_base = f"{base_path}_{timestamp}"
        print(f"Saving model to {save_base}.zip ...", flush=True)
        _model_for_saving.save(save_base)
        print("Model saved.", flush=True)
    except Exception as e:
        print("Failed to save model:", e, flush=True)

def _signal_handler(signum, frame):
    print(f"Received signal {signum}, attempting to save model and exit...", flush=True)
    _safe_save_model(base_path="models/snake_dqn_model_signal")
    os._exit(0)

# register signal handlers and atexit
signal.signal(signal.SIGINT, _signal_handler)   # Ctrl-C
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(lambda: _safe_save_model(base_path="models/snake_dqn_model_atexit"))

class PlottingCallback(BaseCallback):
    """
    Updates live plot and detects pygame window close events.
    If the pygame window is closed, the callback will save the model and stop training.
    """
    def __init__(self, check_freq=5000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self._init_plot = False

    def _init_matplotlib(self):
        import matplotlib.pyplot as plt
        plt.ion()
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(8,4))
        self._init_plot = True

    def _on_training_start(self) -> None:
        # ensure a global model reference is available for saving
        global _model_for_saving
        _model_for_saving = self.model

    def _on_step(self) -> bool:
        # initialize plotting lazily
        if not self._init_plot:
            self._init_matplotlib()

        # update plot periodically
        if self.num_timesteps % self.check_freq == 0:
            self._update_plot()

        # If pygame is used for rendering, poll events and handle window close.
        # Closing the pygame window typically posts a pygame.QUIT event.
        try:
            import pygame
            # quick poll: if pygame not initialized, skip
            if pygame.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Pygame window closed by user. Saving model and stopping training...", flush=True)
                        _safe_save_model(base_path="models/snake_dqn_model_pygame_close")
                        return False  # stop training
        except Exception:
            # pygame not installed or other error -> ignore
            pass

        return True  # continue training

    def _update_plot(self):
        import pandas as pd
        path = "logs/monitor.csv"
        if not os.path.exists(path):
            return
        try:
            df = pd.read_csv(path, skiprows=1)
            rewards = df['r'].values
            if rewards.size == 0:
                return
            mean = pd.Series(rewards).rolling(window=100, min_periods=1).mean().values

            self.ax.clear()
            self.ax.plot(rewards, color='skyblue', label='episode reward')
            self.ax.plot(mean, color='orange', label='100-ep mean')
            self.ax.set_xlabel('Episode')
            self.ax.set_ylabel('Reward')
            self.ax.set_title('Training progress')
            self.ax.legend()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            # ignore transient read/write issues
            pass

def main(total_timesteps=200_000, render=True):
    global _model_for_saving

    render_mode = "human" if render else None
    # create environment; Monitor writes episode rewards to logs/monitor.csv
    env = DummyVecEnv([lambda: Monitor(SnakeGymEnv(render_mode=render_mode), filename="logs/monitor.csv")])

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/",
                learning_rate=1e-3, buffer_size=100_000, batch_size=64, gamma=0.99)

    _model_for_saving = model

    plot_cb = PlottingCallback(check_freq=2000)

    try:
        model.learn(total_timesteps=total_timesteps, callback=plot_cb)
        # normal finish: save final model
        print("Training finished, saving final model...", flush=True)
        model.save("models/snake_dqn_model_final")
        print("Final model saved to models/snake_dqn_model_final.zip", flush=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, saving model before exit...", flush=True)
        _safe_save_model(base_path="models/snake_dqn_model_interrupt")
    except Exception as e:
        print("Exception during training:", e, flush=True)
        print("Attempting to save model before exit...", flush=True)
        _safe_save_model(base_path="models/snake_dqn_model_error")
        raise
    finally:
        # last attempt to save and close env
        _safe_save_model(base_path="models/snake_dqn_model_final_attempt")
        env.close()

if __name__ == "__main__":
    # set render=True to allow pygame window; closing it will trigger save and stop.
    main(total_timesteps=200_000, render=True)
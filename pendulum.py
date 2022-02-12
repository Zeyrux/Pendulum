import pathlib
import os

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

EPISODES = 20
TOTAL_TIMESTEPS = 1000000
LOG_PATH = "Logs"
MODEL_PATH = os.path.join("Model", "model.zip")

env = gym.make("Pendulum-v0")
env = DummyVecEnv([lambda: env])

if not os.path.isdir(LOG_PATH):
    pathlib.Path(LOG_PATH).mkdir(exist_ok=True, parents=True)
if not os.path.isdir(MODEL_PATH[:MODEL_PATH.rindex("\\")]):
    pathlib.Path(MODEL_PATH[:MODEL_PATH.rindex("\\")]).mkdir(exist_ok=True, parents=True)


# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_PATH)
model = PPO.load(MODEL_PATH, env=env, learn_rate=0.1)


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.lr = 0.00001

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        self.lr -= self.lr / 100
        self.model.learning_rate = self.lr
        print(self.model.learning_rate)

    def _on_step(self) -> bool:
        pass

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass


def test_model():
    best_score = [-100, -1]
    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        score = 0
        frames = 0

        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward
            frames += 1
        result = score / frames
        if result > best_score[0]:
            best_score = [result, episode]
        print(f"Episode: {episode}; average Score: {result}")
    print(f"Best Score: {best_score[0]} at episode: {best_score[1]}")


def train():
    callback = CustomCallback()
    for i in range(10):
        print(i)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS/10,
            callback=callback
        )
        model.save(MODEL_PATH)


def main():
    test_model()
    env.close()


if __name__ == "__main__":
    main()

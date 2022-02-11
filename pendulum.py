import pathlib
import os

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

EPISODES = 10
TOTAL_TIMESTEPS = 2000000
LOG_PATH = "Logs"
MODEL_PATH = os.path.join("Model", "model.zip")

env = gym.make("Pendulum-v0")
env = DummyVecEnv([lambda: env])

if not os.path.isdir(LOG_PATH):
    pathlib.Path(LOG_PATH).mkdir(exist_ok=True, parents=True)
if not os.path.isdir(MODEL_PATH[:MODEL_PATH.rindex("\\")]):
    pathlib.Path(MODEL_PATH[:MODEL_PATH.rindex("\\")]).mkdir(exist_ok=True, parents=True)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_PATH)
#model = PPO.load(MODEL_PATH, env=env)


def test_model():
    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward + 1
        print(f"Episode: {episode}; Score: {score}")


def train():
    reduce_lr = 0.1
    model.learning_rate(0.1)
    for i in range(100):
        print(i)
        if i % 10 == 0:
            reduce_lr /= 10
        model.learning_rate(model.learning_rate - reduce_lr)
        model.learn(total_timesteps=TOTAL_TIMESTEPS/100)
        model.save(MODEL_PATH)


def main():
    train()
    env.close()


if __name__ == "__main__":
    main()

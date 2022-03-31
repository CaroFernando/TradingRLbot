import json
from stable_baselines3.common.env_checker import check_env
from env.CryptoEnv import *
from stable_baselines3 import PPO
from models.CustomModel import *
import matplotlib.pyplot as plt

policy_kwargs = dict(
    features_extractor_class=LSTMfeatures,
    features_extractor_kwargs=dict(features_dim=128),
)

env = CustomEnv(32, 'ETHUSDT_TEST')
model = PPO.load('logs/lstmModel_400000_steps')

observation = env.reset()
done = False

budget = []

while not done:
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    print(f"{action} {reward} {done}")

    if not done: budget.append(env.calcTotalBudget(env.data.iloc[env.ind][3]))

env.reset()
plt.plot(budget)
plt.savefig('budget.png')


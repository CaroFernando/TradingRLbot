import json
from stable_baselines3.common.env_checker import check_env
from env.CryptoEnv import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from models.CustomModel import *

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='logs/',name_prefix='lstmModel')

policy_kwargs = dict(
    features_extractor_class=LSTMfeatures,
    features_extractor_kwargs=dict(features_dim=128),
)

env = CustomEnv(32, 'ETHUSDT')
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="TBLog/")
model.learn(total_timesteps=int(1e10) , tb_log_name="lstmrun", callback=checkpoint_callback)
model.save("finalLSTM")
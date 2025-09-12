import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from balance_env_sb3 import BalanceEnv

# 환경 생성
env = DummyVecEnv([lambda: BalanceEnv(render_mode=None)])

# PPO 모델 생성
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)


# 학습
model.learn(total_timesteps=500000)

# 모델 저장
model.save("balance_ppo") 
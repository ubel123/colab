from balance_env_sb3 import BalanceEnv
from stable_baselines3 import PPO
import pygame

# 환경 생성 (학습 때와 action_type 동일해야 함)
env = BalanceEnv(action_type='continuous', render_mode='human')

# 저장된 모델 불러오기
model = PPO.load("balance_ppo")

obs, _ = env.reset()
running = True

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # 학습된 모델의 행동 예측
    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)
    env.render(mode='human')

    if terminated or truncated:
        obs, _ = env.reset()

env.close()


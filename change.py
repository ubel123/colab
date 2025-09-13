import torch
from stable_baselines3 import PPO
import numpy as np

# 모델 로드
model = PPO.load("ppo_js_balance_bot.zip")

# Dummy input (observation space: 4,)
dummy_input = torch.tensor(np.zeros((1, 4), dtype=np.float32))

# 정책 추출
policy = model.policy
policy.eval()

# ONNX로 내보내기
torch.onnx.export(
    policy,
    dummy_input,
    "ppo_js_balance_bot.onnx",
    input_names=["observation"],
    output_names=["action"],
    dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
    opset_version=11
)

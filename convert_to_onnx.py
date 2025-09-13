# filepath: c:\Users\User\Desktop\colab\convert_to_onnx.py
import torch
from stable_baselines3 import PPO

print("ppo_js_balance_bot.zip 파일을 model.onnx로 변환 시작...")
model = PPO.load("ppo_js_balance_bot.zip", device='cpu')
policy = model.policy
policy.eval()
dummy_input = torch.randn(1, 4)
torch.onnx.export(
    policy, dummy_input, "model.onnx", opset_version=11,
    input_names=['observation'], output_names=['action'],
    dynamic_axes={'observation':{0:'batch_size'}, 'action':{0:'batch_size'}}
)
print("성공! 'model.onnx' 파일이 생성되었습니다.")
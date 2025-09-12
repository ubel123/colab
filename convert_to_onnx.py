import torch
import numpy as np
import onnx
import onnxruntime as ort
from stable_baselines3 import PPO

# ------------------------------
# 1. 환경 불러오기
# ------------------------------
from balance_env_sb3 import BalanceEnv  # environment.py 안에 BalanceEnv 클래스가 정의되어 있어야 함

# ------------------------------
# 2. 학습된 SB3 모델 로드
# ------------------------------
model = PPO.load("balance_ppo.zip")

# ------------------------------
# 3. 환경 인스턴스 생성 후 관측 공간 확인
# ------------------------------
env = BalanceEnv(render_mode=None)
obs_space = env.observation_space

# 더미 입력 (환경 관측값과 동일한 shape)
dummy_obs = torch.tensor(
    obs_space.sample()[None, :],  # 배치 차원 추가
    dtype=torch.float32
)

# ------------------------------
# 4. ONNX 변환
# ------------------------------
onnx_path = "model.onnx"
torch.onnx.export(
    model.policy,              # SB3 모델 내부의 PyTorch policy
    dummy_obs,                 # 더미 입력
    onnx_path,                 # 저장 파일명
    input_names=['observation'],
    output_names=['action'],
    opset_version=12,
    dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
)

print(f"✅ ONNX 모델이 {onnx_path} 로 변환되었습니다.")

# ------------------------------
# 5. ONNX Runtime 검증
# ------------------------------
# 변환된 ONNX 모델 로드 및 유효성 검사
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# ONNX Runtime 세션 생성
ort_session = ort.InferenceSession(onnx_path)

# 입력값 준비 (numpy 형식)
dummy_input_np = dummy_obs.detach().cpu().numpy().astype(np.float32)
outputs = ort_session.run(None, {"observation": dummy_input_np})

print("✅ ONNX Runtime 추론 결과:")
print(outputs[0])

# filepath: c:\Users\User\Desktop\colab\train_js.py
import gymnasium as gym, numpy as np, asyncio, websockets, json, webbrowser, os
from gymnasium import spaces
from stable_baselines3 import PPO

class JavaScriptEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._ensure_server_and_connection())

    async def _ensure_server_and_connection(self):
        if not hasattr(self, 'server'):
            # --- 여기가 수정되었습니다 ---
            # self._handler를 직접 전달하는 대신, 올바른 인자를 받는 람다 함수를 사용합니다.
            async def handler_wrapper(websocket):
                await self._handler(websocket, path=None) # path 인자를 명시적으로 전달

            self.server = await websockets.serve(handler_wrapper, "localhost", 8765)
            webbrowser.open(f'file://{os.path.realpath("index.html")}')
        if not hasattr(self, 'websocket'):
            print("브라우저 클라이언트의 연결을 기다립니다...")
            self.conn_queue = asyncio.Queue(1)
            await self.conn_queue.get()
            print("클라이언트가 성공적으로 연결되었습니다.")

    # _handler 함수는 그대로 유지합니다.
    async def _handler(self, websocket, path):
        if hasattr(self, 'websocket') and self.websocket: return
        self.websocket = websocket
        await self.conn_queue.put(True)
        try:
            await websocket.wait_closed()
        finally:
            self.websocket = None

    async def _send_recv(self, cmd):
        await self.websocket.send(json.dumps(cmd))
        return json.loads(await self.websocket.recv())

    def step(self, action):
        res = self.loop.run_until_complete(self._send_recv({'command':'step','action':int(action)}))
        return np.array(res['observation'],dtype=np.float32),res['reward'],res['terminated'],res['truncated'],{}

    def reset(self, seed=None, options=None):
        res = self.loop.run_until_complete(self._send_recv({'command':'reset'}))
        return np.array(res['observation'],dtype=np.float32), {}

    def close(self):
        if hasattr(self, 'server'): self.server.close()

if __name__ == "__main__":
    env = JavaScriptEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0)
    model.learn(total_timesteps=700000, progress_bar=True)
    model.save("ppo_js_balance_bot")
    print("\n학습 완료! 'ppo_js_balance_bot.zip' 파일 저장됨.")
    env.close()
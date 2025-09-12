import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pymunk
import pygame

class BalanceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, width=600, height=400,
                 action_type='continuous', max_force=15000.0,
                 dt=1/50.0, max_steps=1000, render_mode=None):
        super().__init__()
        self.width = width
        self.height = height
        self.action_type = action_type
        self.max_force = max_force
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Action space
        if action_type == 'continuous':
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)

        # Observation space: normalized
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Physics
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -981.0)
        self._init_physics()

        # Rendering
        self.screen = None
        self.clock = None
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        # Episode bookkeeping
        self.steps = 0
        self._max_angle = math.radians(12.0)
        self.seed()
        self._last_rgb = None

    def _init_physics(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -981.0)
        self.static = self.space.static_body

        # Cart
        cart_w, cart_h = 60.0, 30.0
        cart_mass = 1.0
        cart_moment = pymunk.moment_for_box(cart_mass, (cart_w, cart_h))
        self.cart_body = pymunk.Body(cart_mass, cart_moment)
        self.cart_body.position = (self.width / 2.0, 120.0)
        self.cart_shape = pymunk.Poly.create_box(self.cart_body, (cart_w, cart_h))
        self.cart_shape.friction = 0.8
        self.space.add(self.cart_body, self.cart_shape)

        # GrooveJoint
        groove_y = 120.0
        g = pymunk.GrooveJoint(self.static, self.cart_body,
                               (50.0, groove_y), (self.width - 50.0, groove_y),
                               (0, 0))
        self.space.add(g)

        # Pole
        pole_length = 120.0
        pole_w = 6.0
        pole_mass = 0.1
        pole_moment = pymunk.moment_for_box(pole_mass, (pole_w, pole_length))
        self.pole_body = pymunk.Body(pole_mass, pole_moment)
        self.pole_body.position = (self.cart_body.position.x,
                                   self.cart_body.position.y + pole_length / 2.0 + cart_h / 2.0)
        self.pole_shape = pymunk.Poly.create_box(self.pole_body, (pole_w, pole_length))
        self.pole_shape.friction = 0.1
        self.space.add(self.pole_body, self.pole_shape)

        # PinJoint
        pivot = pymunk.PinJoint(self.cart_body, self.pole_body,
                                (0, cart_h / 2.0), (0, -pole_length / 2.0))
        self.space.add(pivot)

        # Damping
        self.cart_body.velocity_func = lambda body, gravity, damping, dt: pymunk.Body.update_velocity(body, gravity, 0.999, dt)
        self.pole_body.angular_velocity *= 0.0

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    def _get_observation(self):
        cart_x = self.cart_body.position.x
        cart_vx = self.cart_body.velocity.x
        vec = self.pole_body.position - self.cart_body.position
        world_angle = math.atan2(vec.y, vec.x)
        pole_angle = world_angle - (math.pi / 2.0)
        pole_ang_vel = self.pole_body.angular_velocity

        obs = np.array([
            (cart_x - self.width/2.0) / (self.width/2.0),
            np.clip(cart_vx / 200.0, -1.0, 1.0),
            np.clip(pole_angle / self._max_angle, -1.0, 1.0),
            np.clip(pole_ang_vel / 10.0, -1.0, 1.0)
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if self.action_type == 'continuous':
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)[0]
            force = float(a) * self.max_force
        else:
            act = int(action[0]) if isinstance(action, (np.ndarray, list)) else int(action)
            force = {0: -self.max_force, 1: 0.0, 2: self.max_force}[act]

        self.cart_body.apply_force_at_local_point((force, 0.0), (0.0, 0.0))

        # Physics substeps
        for _ in range(4):
            self.space.step(self.dt / 4)

        self.steps += 1
        obs = self._get_observation()

        # Termination
        angle = obs[2] * self._max_angle
        terminated = bool(abs(angle) > self._max_angle)
        truncated = bool(self.steps >= self.max_steps)

        angle_penalty = (abs(angle) / self._max_angle) ** 2
        cart_dist_penalty = abs((self.cart_body.position.x - self.width/2.0) / (self.width/2.0))
        reward = 1.0 - angle_penalty - 0.1 * cart_dist_penalty
        if terminated:
            reward = -1.0

        return obs, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.space.remove(*self.space.bodies, *self.space.shapes, *self.space.constraints)
        self._init_physics()
        self.steps = 0

        self.cart_body.position = (self.width/2.0 + self.np_random.uniform(-5,5), 120.0)
        self.cart_body.velocity = (self.np_random.uniform(-5,5), 0.0)
        self.pole_body.position = (self.cart_body.position.x, self.cart_body.position.y + 80.0)
        self.pole_body.angle = self.np_random.uniform(-0.05, 0.05)
        self.pole_body.angular_velocity = self.np_random.uniform(-0.1, 0.1)

        return self._get_observation(), {}

    def _to_pygame(self, p):
        return int(p[0]), int(self.height - p[1])

    def render(self, mode='human'):
        surface = pygame.Surface((self.width, self.height))
        surface.fill((200, 200, 200))
        pygame.draw.line(surface, (50,50,50), (50, self.height-120), (self.width-50, self.height-120), 4)

        cart_pos = self._to_pygame(self.cart_body.position)
        cart_w, cart_h = 60, 30
        cart_rect = pygame.Rect(0,0,cart_w,cart_h)
        cart_rect.center = cart_pos
        pygame.draw.rect(surface, (20,120,200), cart_rect)

        pivot_px = self._to_pygame(self.cart_body.position + (0,15))
        vec = self.pole_body.position - self.cart_body.position
        tip_px = self._to_pygame((self.pole_body.position.x + vec.x, self.pole_body.position.y + vec.y))
        pygame.draw.line(surface, (200,50,50), pivot_px, tip_px, 6)

        if mode == "human":
            if self.screen is None:
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
            self.screen.blit(surface, (0,0))
            pygame.display.flip()
            if self.clock:
                self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            arr = pygame.surfarray.array3d(surface)
            return np.transpose(arr, (1,0,2))

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

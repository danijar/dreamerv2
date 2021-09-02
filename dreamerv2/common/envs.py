import os
import threading

import gym
import numpy as np


class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    os.environ['MUJOCO_GL'] = 'egl'
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def step(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky_actions else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._env = env
    self._grayscale = grayscale

  @property
  def observation_space(self):
    return gym.spaces.Dict({
        'image': self._env.observation_space,
        'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
    })

  @property
  def action_space(self):
    return gym.spaces.Dict({'action': self._env.action_space})

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs

  def step(self, action):
    action = action['action']
    image, reward, done, info = self._env.step(action)
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs, reward, done, info

  def render(self, mode):
    return self._env.render(mode)


class Crafter:

  def __init__(self, outdir=None, seed=None):
    import crafter
    self._env = crafter.Env(seed=seed)
    if outdir:
      self._env = crafter.Recorder(
          self._env, outdir,
          save_episode=True,
          save_video=False,
          include_image=False,
      )
    self._achievements = crafter.constants.achievements.copy()

  @property
  def observation_space(self):
    spaces = {'image': self._env.observation_space}
    for name in self._achievements:
      spaces[name] = gym.spaces.Box(0, 2 ** 31, (), np.int32)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return gym.spaces.Dict({'action': self._env.action_space})

  def close(self):
    pass

  def reset(self):
    image = self._env.reset()
    obs = {'image': image}
    obs.update({
        f'log_achievement_{name}': 0
        for name in self._achievements})
    return obs

  def step(self, action):
    action = action['action']
    image, reward, done, info = self._env.step(action)
    obs = {'image': image}
    obs.update({
        f'log_achievement_{key}': value
        for key, value in info['achievements'].items()})
    return obs, reward, done, info

  def render(self, mode):
    return self._env.render()


class Dummy:

  def __init__(self):
    pass

  @property
  def observation_space(self):
    image = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
    return gym.spaces.Dict({'image': image})

  @property
  def action_space(self):
    action = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def step(self, action):
    obs = {'image': np.zeros((64, 64, 3))}
    reward = 0.0
    done = False
    info = {}
    return obs, reward, done, info

  def reset(self):
    obs = {'image': np.zeros((64, 64, 3))}
    return obs


class DictSpaces:

  def __init__(self, env, obs_key='image', act_key='action'):
    self._env = env
    if hasattr(env.observation_space, 'spaces'):
      obs_key = False
    if hasattr(env.action_space, 'spaces'):
      act_key = False
    self._obs_key = obs_key
    self._act_key = act_key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = self._env.observation_space
    if self._obs_key:
      space = gym.spaces.Dict({self._obs_key: space})
    return space

  @property
  def action_space(self):
    space = self._env.action_space
    if self._act_key:
      space = gym.spaces.Dict({self._act_key: space})
    return space

  def step(self, action):
    if self._act_key:
      action = action[self._act_key]
    obs, reward, done, info = self._env.step(action)
    if self._obs_key:
      obs = {self._obs_key: obs}
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    if self._obs_key:
      obs = {self._obs_key: obs}
    return obs


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.action_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class OneHotAction:

  def __init__(self, env, key='action'):
    assert hasattr(env.action_space[key], 'n')
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

  def step(self, action):
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index})

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class ResizeImage:

  def __init__(self, env, size=(64, 64), key='image'):
    assert key in env.observation_space.spaces
    self._env = env
    self._key = key
    self._shape = size

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    shape = self._shape + spaces[self._key].shape[2:]
    spaces[self._key] = gym.spaces.Box(0, 255, shape, np.uint8)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs[self._key] = self._resize(obs[self._key])
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs[self._key] = self._resize(obs[self._key])
    return obs

  def _resize(self, image):
    from PIL import Image
    image = Image.fromarray(image)
    image = image.resize(self._shape, Image.NEAREST)
    image = np.array(image)
    return image


class RewardObs:

  def __init__(self, env, key='reward'):
    assert key not in env.observation_space.spaces
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
    return gym.spaces.Dict({
        **self._env.observation_space.spaces, self._key: space})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class ResetObs:

  def __init__(self, env, key='reset'):
    assert key not in env.observation_space.spaces
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = gym.spaces.Box(0, 1, (), dtype=np.bool)
    return gym.spaces.Dict({
        **self._env.observation_space.spaces, self._key: space})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reset'] = np.array(False, np.bool)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reset'] = np.array(True, np.bool)
    return obs

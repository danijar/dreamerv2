import numpy as np


class Driver:

  def __init__(self, envs, **kwargs):
    self._envs = envs
    self._kwargs = kwargs
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._actspaces = [env.action_space.spaces for env in envs]
    self.reset()

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._dones = [True] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      for i, done in enumerate(self._dones):
        if done:
          self._obs[i] = ob = self._envs[i].reset()
          act = {k: np.zeros(v.shape) for k, v in self._actspaces[i].items()}
          tran = {**ob, **act, 'reward': 0.0, 'discount': 1.0, 'done': False}
          [callback(tran, **self._kwargs) for callback in self._on_resets]
          self._eps[i] = [tran]
      obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
      actions, self._state = policy(obs, self._state, **self._kwargs)
      actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
      assert len(actions) == len(self._envs)
      results = [e.step(a) for e, a in zip(self._envs, actions)]
      for i, (act, (ob, rew, done, info)) in enumerate(zip(actions, results)):
        obs = {k: self._convert(v) for k, v in obs.items()}
        disc = info.get('discount', np.array(1 - float(done)))
        tran = {**ob, **act, 'reward': rew, 'discount': disc, 'done': done}
        [callback(tran, **self._kwargs) for callback in self._on_steps]
        self._eps[i].append(tran)
        if done:
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          [callback(ep, **self._kwargs) for callback in self._on_episodes]
      obs, _, dones = zip(*[p[:3] for p in results])
      self._obs = list(obs)
      self._dones = list(dones)
      episode += sum(dones)
      step += len(dones)

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value

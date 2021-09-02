import re

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import dists


class AttrDict(dict):

  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


class RandomAgent:

  def __init__(self, action_space, logprob=False):
    self._logprob = logprob
    if hasattr(action_space, 'n'):
      self._dist = dists.OneHotDist(tf.zeros(action_space.n))
    else:
      dist = tfd.Uniform(action_space.low, action_space.high)
      self._dist = tfd.Independent(dist, 1)

  def __call__(self, obs, state=None, mode=None):
    action = self._dist.sample(len(obs['reset']))
    output = {'action': action}
    if self._logprob:
      output['logprob'] = self._dist.log_prob(action)
    return output, None


def static_scan(fn, inputs, start, reverse=False):
  last = start
  outputs = [[] for _ in tf.nest.flatten(start)]
  indices = range(tf.nest.flatten(inputs)[0].shape[0])
  if reverse:
    indices = reversed(indices)
  for index in indices:
    inp = tf.nest.map_structure(lambda x: x[index], inputs)
    last = fn(last, inp)
    [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [tf.stack(x, 0) for x in outputs]
  return tf.nest.pack_sequence_as(start, outputs)


def schedule(string, step):
  try:
    return float(string)
  except ValueError:
    step = tf.cast(step, tf.float32)
    match = re.match(r'linear\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, duration = [float(group) for group in match.groups()]
      mix = tf.clip_by_value(step / duration, 0, 1)
      return (1 - mix) * initial + mix * final
    match = re.match(r'warmup\((.+),(.+)\)', string)
    if match:
      warmup, value = [float(group) for group in match.groups()]
      scale = tf.clip_by_value(step / warmup, 0, 1)
      return scale * value
    match = re.match(r'exp\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, halflife = [float(group) for group in match.groups()]
      return (initial - final) * 0.5 ** (step / halflife) + final
    match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, duration = [float(group) for group in match.groups()]
      mix = tf.clip_by_value(step / duration, 0, 1)
      horizon = (1 - mix) * initial + mix * final
      return 1 - 1 / horizon
    raise NotImplementedError(string)


def lambda_return(
    reward, value, pcont, bootstrap, lambda_, axis):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
  if isinstance(pcont, (int, float)):
    pcont = pcont * tf.ones_like(reward)
  dims = list(range(reward.shape.ndims))
  dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
  if axis != 0:
    reward = tf.transpose(reward, dims)
    value = tf.transpose(value, dims)
    pcont = tf.transpose(pcont, dims)
  if bootstrap is None:
    bootstrap = tf.zeros_like(value[-1])
  next_values = tf.concat([value[1:], bootstrap[None]], 0)
  inputs = reward + pcont * next_values * (1 - lambda_)
  returns = static_scan(
      lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
      (inputs, pcont), bootstrap, reverse=True)
  if axis != 0:
    returns = tf.transpose(returns, dims)
  return returns


def action_noise(action, amount, action_space):
  if amount == 0:
    return action
  amount = tf.cast(amount, action.dtype)
  if hasattr(action_space, 'n'):
    probs = amount / action.shape[-1] + (1 - amount) * action
    return dists.OneHotDist(probs=probs).sample()
  else:
    return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)


def pad_dims(tensor, total_dims):
  while len(tensor.shape) < total_dims:
    tensor = tensor[..., None]
  return tensor

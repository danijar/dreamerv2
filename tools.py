import datetime
import io
import json
import pathlib
import pickle
import re
import time
import uuid

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd


# Patch to ignore seed to avoid synchronization across GPUs.
_orig_random_categorical = tf.random.categorical
def random_categorical(*args, **kwargs):
  kwargs['seed'] = None
  return _orig_random_categorical(*args, **kwargs)
tf.random.categorical = random_categorical

# Patch to ignore seed to avoid synchronization across GPUs.
_orig_random_normal = tf.random.normal
def random_normal(*args, **kwargs):
  kwargs['seed'] = None
  return _orig_random_normal(*args, **kwargs)
tf.random.normal = random_normal


class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


class Module(tf.Module):

  def save(self, filename):
    values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f'Save checkpoint with {amount} tensors and {count} parameters.')
    with pathlib.Path(filename).open('wb') as f:
      pickle.dump(values, f)

  def load(self, filename):
    with pathlib.Path(filename).open('rb') as f:
      values = pickle.load(f)
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f'Load checkpoint with {amount} tensors and {count} parameters.')
    tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

  def get(self, name, ctor, *args, **kwargs):
    # Create or get layer by name to avoid mentioning it in the constructor.
    if not hasattr(self, '_modules'):
      self._modules = {}
    if name not in self._modules:
      self._modules[name] = ctor(*args, **kwargs)
    return self._modules[name]


def var_nest_names(nest):
  if isinstance(nest, dict):
    items = ' '.join(f'{k}:{var_nest_names(v)}' for k, v in nest.items())
    return '{' + items + '}'
  if isinstance(nest, (list, tuple)):
    items = ' '.join(var_nest_names(v) for v in nest)
    return '[' + items + ']'
  if hasattr(nest, 'name') and hasattr(nest, 'shape'):
    return nest.name + str(nest.shape).replace(', ', 'x')
  if hasattr(nest, 'shape'):
    return str(nest.shape).replace(', ', 'x')
  return '?'


class Logger:

  def __init__(self, logdir, step):
    self._logdir = logdir
    self._writer = tf.summary.create_file_writer(str(logdir), max_queue=1000)
    self._last_step = None
    self._last_time = None
    self._scalars = {}
    self._images = {}
    self._videos = {}
    self.step = step

  def scalar(self, name, value):
    self._scalars[name] = float(value)

  def image(self, name, value):
    self._images[name] = np.array(value)

  def video(self, name, value):
    self._videos[name] = np.array(value)

  def write(self, fps=False):
    scalars = list(self._scalars.items())
    if fps:
      scalars.append(('fps', self._compute_fps(self.step)))
    print(f'[{self.step}]', ' / '.join(f'{k} {v:.1f}' for k, v in scalars))
    with (self._logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': self.step, ** dict(scalars)}) + '\n')
    with self._writer.as_default():
      for name, value in scalars:
        tf.summary.scalar('scalars/' + name, value, self.step)
      for name, value in self._images.items():
        tf.summary.image(name, value, self.step)
      for name, value in self._videos.items():
        video_summary(name, value, self.step)
    self._writer.flush()
    self._scalars = {}
    self._images = {}
    self._videos = {}

  def _compute_fps(self, step):
    if self._last_step is None:
      self._last_time = time.time()
      self._last_step = step
      return 0
    steps = step - self._last_step
    duration = time.time() - self._last_time
    self._last_time += duration
    self._last_step = step
    return steps / duration


def graph_summary(writer, step, fn, *args):
  def inner(*args):
    tf.summary.experimental.set_step(step.numpy().item())
    with writer.as_default():
      fn(*args)
  return tf.numpy_function(inner, args, [])


def video_summary(name, video, step=None, fps=20):
  name = name if isinstance(name, str) else name.decode('utf-8')
  if np.issubdtype(video.dtype, np.floating):
    video = np.clip(255 * video, 0, 255).astype(np.uint8)
  B, T, H, W, C = video.shape
  try:
    frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    summary = tf1.Summary()
    image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
    image.encoded_image_string = encode_gif(frames, fps)
    summary.value.add(tag=name, image=image)
    tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
  except (IOError, OSError) as e:
    print('GIF summaries require ffmpeg in $PATH.', e)
    frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
    tf.summary.image(name, frames, step)


def encode_gif(frames, fps):
  from subprocess import Popen, PIPE
  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  cmd = ' '.join([
      f'ffmpeg -y -f rawvideo -vcodec rawvideo',
      f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
      f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
      f'-r {fps:.02f} -f gif -'])
  proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in frames:
    proc.stdin.write(image.tostring())
  out, err = proc.communicate()
  if proc.returncode:
    raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
  del proc
  return out


def simulate(agent, envs, steps=0, episodes=0, state=None):
  # Initialize or unpack simulation state.
  if state is None:
    step, episode = 0, 0
    done = np.ones(len(envs), np.bool)
    length = np.zeros(len(envs), np.int32)
    obs = [None] * len(envs)
    agent_state = None
  else:
    step, episode, done, length, obs, agent_state = state
  while (steps and step < steps) or (episodes and episode < episodes):
    # Reset envs if necessary.
    if done.any():
      indices = [index for index, d in enumerate(done) if d]
      results = [envs[i].reset() for i in indices]
      for index, result in zip(indices, results):
        obs[index] = result
    # Step agents.
    obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    action, agent_state = agent(obs, done, agent_state)
    if isinstance(action, dict):
      action = [
          {k: np.array(action[k][i]) for k in action}
          for i in range(len(envs))]
    else:
      action = np.array(action)
    assert len(action) == len(envs)
    # Step envs.
    results = [e.step(a) for e, a in zip(envs, action)]
    obs, _, done = zip(*[p[:3] for p in results])
    obs = list(obs)
    done = np.stack(done)
    episode += int(done.sum())
    length += 1
    step += (done * length).sum()
    length *= (1 - done)
  # Return new state to allow resuming the simulation.
  return (step - steps, episode - episodes, done, length, obs, agent_state)


def save_episodes(directory, episodes):
  directory = pathlib.Path(directory).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  filenames = []
  for episode in episodes:
    identifier = str(uuid.uuid4().hex)
    length = len(episode['reward'])
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
      np.savez_compressed(f1, **episode)
      f1.seek(0)
      with filename.open('wb') as f2:
        f2.write(f1.read())
    filenames.append(filename)
  return filenames


def sample_episodes(episodes, length=None, balance=False, seed=0):
  random = np.random.RandomState(seed)
  while True:
    episode = random.choice(list(episodes.values()))
    if length:
      total = len(next(iter(episode.values())))
      available = total - length
      if available < 1:
        print(f'Skipped short episode of length {available}.')
        continue
      if balance:
        index = min(random.randint(0, total), available)
      else:
        index = int(random.randint(0, available + 1))
      episode = {k: v[index: index + length] for k, v in episode.items()}
    yield episode


def load_episodes(directory, limit=None):
  directory = pathlib.Path(directory).expanduser()
  episodes = {}
  total = 0
  for filename in reversed(sorted(directory.glob('*.npz'))):
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode: {e}')
      continue
    episodes[str(filename)] = episode
    total += len(episode['reward']) - 1
    if limit and total >= limit:
      break
  return episodes


class DtypeDist:

  def __init__(self, dist, dtype=None):
    self._dist = dist
    self._dtype = dtype or prec.global_policy().compute_dtype

  @property
  def name(self):
    return 'DtypeDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    return tf.cast(self._dist.mean(), self._dtype)

  def mode(self):
    return tf.cast(self._dist.mode(), self._dtype)

  def entropy(self):
    return tf.cast(self._dist.entropy(), self._dtype)

  def sample(self, *args, **kwargs):
    return tf.cast(self._dist.sample(*args, **kwargs), self._dtype)


class SampleDist:

  def __init__(self, dist, samples=100):
    self._dist = dist
    self._samples = samples

  @property
  def name(self):
    return 'SampleDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    samples = self._dist.sample(self._samples)
    return tf.reduce_mean(samples, 0)

  def mode(self):
    sample = self._dist.sample(self._samples)
    logprob = self._dist.log_prob(sample)
    return tf.gather(sample, tf.argmax(logprob))[0]

  def entropy(self):
    sample = self._dist.sample(self._samples)
    logprob = self.log_prob(sample)
    return -tf.reduce_mean(logprob, 0)


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=None):
    self._sample_dtype = dtype or prec.global_policy().compute_dtype
    super().__init__(logits=logits, probs=probs)

  def mode(self):
    return tf.cast(super().mode(), self._sample_dtype)

  def sample(self, sample_shape=(), seed=None):
    # Straight through biased gradient estimator.
    sample = tf.cast(super().sample(sample_shape, seed), self._sample_dtype)
    probs = super().probs_parameter()
    while len(probs.shape) < len(sample.shape):
      probs = probs[None]
    sample += tf.cast(probs - tf.stop_gradient(probs), self._sample_dtype)
    return sample


class GumbleDist(tfd.RelaxedOneHotCategorical):

  def __init__(self, temp, logits=None, probs=None, dtype=None):
    self._sample_dtype = dtype or prec.global_policy().compute_dtype
    self._exact = tfd.OneHotCategorical(logits=logits, probs=probs)
    super().__init__(temp, logits=logits, probs=probs)

  def mode(self):
    return tf.cast(self._exact.mode(), self._sample_dtype)

  def entropy(self):
    return tf.cast(self._exact.entropy(), self._sample_dtype)

  def sample(self, sample_shape=(), seed=None):
    return tf.cast(super().sample(sample_shape, seed), self._sample_dtype)


class UnnormalizedHuber(tfd.Normal):

  def __init__(self, loc, scale, threshold=1, **kwargs):
    self._threshold = tf.cast(threshold, loc.dtype)
    super().__init__(loc, scale, **kwargs)

  def log_prob(self, event):
    return -(tf.math.sqrt(
        (event - self.mean()) ** 2 + self._threshold ** 2) - self._threshold)


class SafeTruncatedNormal(tfd.TruncatedNormal):

  def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
    super().__init__(loc, scale, low, high)
    self._clip = clip
    self._mult = mult

  def sample(self, *args, **kwargs):
    event = super().sample(*args, **kwargs)
    if self._clip:
      clipped = tf.clip_by_value(
          event, self.low + self._clip, self.high - self._clip)
      event = event - tf.stop_gradient(event) + tf.stop_gradient(clipped)
    if self._mult:
      event *= self._mult
    return event


class TanhBijector(tfp.bijectors.Bijector):

  def __init__(self, validate_args=False, name='tanh'):
    super().__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    return tf.nn.tanh(x)

  def _inverse(self, y):
    dtype = y.dtype
    y = tf.cast(y, tf.float32)
    y = tf.where(
        tf.less_equal(tf.abs(y), 1.),
        tf.clip_by_value(y, -0.99999997, 0.99999997), y)
    y = tf.atanh(y)
    y = tf.cast(y, dtype)
    return y

  def _forward_log_det_jacobian(self, x):
    log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
    return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


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


class Optimizer(tf.Module):

  def __init__(
      self, name, lr, eps=1e-4, clip=None, wd=None, wd_pattern=r'.*',
      opt='adam'):
    assert 0 <= wd < 1
    assert not clip or 1 <= clip
    self._name = name
    self._clip = clip
    self._wd = wd
    self._wd_pattern = wd_pattern
    self._opt = {
        'adam': lambda: tf.optimizers.Adam(lr, epsilon=eps),
        'nadam': lambda: tf.optimizers.Nadam(lr, epsilon=eps),
        'adamax': lambda: tf.optimizers.Adamax(lr, epsilon=eps),
        'sgd': lambda: tf.optimizers.SGD(lr),
        'momentum': lambda: tf.optimizers.SGD(lr, 0.9),
    }[opt]()
    self._mixed = (prec.global_policy().compute_dtype == tf.float16)
    if self._mixed:
      self._opt = prec.LossScaleOptimizer(self._opt, 'dynamic')

  @property
  def variables(self):
    return self._opt.variables()

  def __call__(self, tape, loss, modules):
    assert loss.dtype is tf.float32, self._name
    modules = modules if hasattr(modules, '__len__') else (modules,)
    varibs = tf.nest.flatten([module.variables for module in modules])
    count = sum(np.prod(x.shape) for x in varibs)
    print(f'Found {count} {self._name} parameters.')
    assert len(loss.shape) == 0, loss.shape
    tf.debugging.check_numerics(loss, self._name + '_loss')
    if self._mixed:
      with tape:
        loss = self._opt.get_scaled_loss(loss)
    grads = tape.gradient(loss, varibs)
    if self._mixed:
      grads = self._opt.get_unscaled_gradients(grads)
    norm = tf.linalg.global_norm(grads)
    if not self._mixed:
      tf.debugging.check_numerics(norm, self._name + '_norm')
    if self._clip:
      grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
    if self._wd:
      self._apply_weight_decay(varibs)
    self._opt.apply_gradients(zip(grads, varibs))
    metrics = {}
    metrics[f'{self._name}_loss'] = loss
    metrics[f'{self._name}_grad_norm'] = norm
    if self._mixed:
      try:
        metrics[f'{self._name}_loss_scale'] = float(self._opt.loss_scale)
      except TypeError:
        metrics[f'{self._name}_loss_scale'] = float(
            self._opt.loss_scale._current_loss_scale)
    return metrics

  def _apply_weight_decay(self, varibs):
    nontrivial = (self._wd_pattern != r'.*')
    if nontrivial:
      print('Applied weight decay to variables:')
    for var in varibs:
      if re.search(self._wd_pattern, self._name + '/' + var.name):
        if nontrivial:
          print('- ' + self._name + '/' + var.name)
        var.assign((1 - self._wd) * var)


def args_type(default):
  def parse_string(x):
    if default is None:
      return x
    if isinstance(default, bool):
      return bool(['False', 'True'].index(x))
    if isinstance(default, int):
      return float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, (list, tuple)):
      return tuple(args_type(default[0])(y) for y in x.split(','))
    return type(default)(x)
  def parse_object(x):
    if isinstance(default, (list, tuple)):
      return tuple(x)
    return x
  return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start, reverse=False):
  last = start
  outputs = [[] for _ in tf.nest.flatten(start)]
  indices = range(len(tf.nest.flatten(inputs)[0]))
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


def uniform_mixture(dist, dtype=None):
  if dist.batch_shape[-1] == 1:
    return tfd.BatchReshape(dist, dist.batch_shape[:-1])
  dtype = dtype or prec.global_policy().compute_dtype
  weights = tfd.Categorical(tf.zeros(dist.batch_shape, dtype))
  return tfd.MixtureSameFamily(weights, dist)


def cat_mixture_entropy(dist):
  if isinstance(dist, tfd.MixtureSameFamily):
    probs = dist.components_distribution.probs_parameter()
  else:
    probs = dist.probs_parameter()
  return -tf.reduce_mean(
      tf.reduce_mean(probs, 2) *
      tf.math.log(tf.reduce_mean(probs, 2) + 1e-8), -1)


@tf.function
def cem_planner(
    state, num_actions, horizon, proposals, topk, iterations, imagine,
    objective):
  dtype = prec.global_policy().compute_dtype
  B, P = list(state.values())[0].shape[0], proposals
  H, A = horizon, num_actions
  flat_state = {k: tf.repeat(v, P, 0) for k, v in state.items()}
  mean = tf.zeros((B, H, A), dtype)
  std = tf.ones((B, H, A), dtype)
  for _ in range(iterations):
    proposals = tf.random.normal((B, P, H, A), dtype=dtype)
    proposals = proposals * std[:, None] + mean[:, None]
    proposals = tf.clip_by_value(proposals, -1, 1)
    flat_proposals = tf.reshape(proposals, (B * P, H, A))
    states = imagine(flat_proposals, flat_state)
    scores = objective(states)
    scores = tf.reshape(tf.reduce_sum(scores, -1), (B, P))
    _, indices = tf.math.top_k(scores, topk, sorted=False)
    best = tf.gather(proposals, indices, axis=1, batch_dims=1)
    mean, var = tf.nn.moments(best, 1)
    std = tf.sqrt(var + 1e-6)
  return mean[:, 0, :]


@tf.function
def grad_planner(
    state, num_actions, horizon, proposals, iterations, imagine, objective,
    kl_scale, step_size):
  dtype = prec.global_policy().compute_dtype
  B, P = list(state.values())[0].shape[0], proposals
  H, A = horizon, num_actions
  flat_state = {k: tf.repeat(v, P, 0) for k, v in state.items()}
  mean = tf.zeros((B, H, A), dtype)
  rawstd = 0.54 * tf.ones((B, H, A), dtype)
  for _ in range(iterations):
    proposals = tf.random.normal((B, P, H, A), dtype=dtype)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(mean)
      tape.watch(rawstd)
      std = tf.nn.softplus(rawstd)
      proposals = proposals * std[:, None] + mean[:, None]
      proposals = (
          tf.stop_gradient(tf.clip_by_value(proposals, -1, 1)) +
          proposals - tf.stop_gradient(proposals))
      flat_proposals = tf.reshape(proposals, (B * P, H, A))
      states = imagine(flat_proposals, flat_state)
      scores = objective(states)
      scores = tf.reshape(tf.reduce_sum(scores, -1), (B, P))
      div = tfd.kl_divergence(
          tfd.Normal(mean, std),
          tfd.Normal(tf.zeros_like(mean), tf.ones_like(std)))
      elbo = tf.reduce_sum(scores) - kl_scale * div
      elbo /= tf.cast(tf.reduce_prod(tf.shape(scores)), dtype)
    grad_mean, grad_rawstd = tape.gradient(elbo, [mean, rawstd])
    e, v = tf.nn.moments(grad_mean, [1, 2], keepdims=True)
    grad_mean /= tf.sqrt(e * e + v + 1e-4)
    e, v = tf.nn.moments(grad_rawstd, [1, 2], keepdims=True)
    grad_rawstd /= tf.sqrt(e * e + v + 1e-4)
    mean = tf.clip_by_value(mean + step_size * grad_mean, -1, 1)
    rawstd = rawstd + step_size * grad_rawstd
  return mean[:, 0, :]


class Every:

  def __init__(self, every):
    self._every = every
    self._last = None

  def __call__(self, step):
    if not self._every:
      return False
    if self._last is None:
      self._last = step
      return True
    if step >= self._last + self._every:
      self._last += self._every
      return True
    return False


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class Until:

  def __init__(self, until):
    self._until = until

  def __call__(self, step):
    if not self._until:
      return True
    return step < self._until


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

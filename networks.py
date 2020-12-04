import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools


class RSSM(tools.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
      rec_depth=1, shared=False, discrete=False, act=tf.nn.elu,
      mean_act='none', std_act='softplus', temp_post=True, min_std=0.1,
      cell='keras'):
    super().__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._rec_depth = rec_depth
    self._shared = shared
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._temp_post = temp_post
    self._embed = None
    if cell == 'gru':
      self._cell = tfkl.GRUCell(self._deter)
    elif cell == 'gru_layer_norm':
      self._cell = GRUCell(self._deter, norm=True)
    else:
      raise NotImplementedError(cell)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    return state

  @tf.function
  def observe(self, embed, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed, action = swap(embed), swap(action)
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = tools.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state, dtype=None):
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(tools.OneHotDist(logit), 1)
      if dtype != tf.float32:
        dist = tools.DtypeDist(dist, dtype or state['logit'].dtype)
    else:
      mean, std = state['mean'], state['std']
      if dtype:
        mean = tf.cast(mean, dtype)
        std = tf.cast(std, dtype)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    if not self._embed:
      self._embed = embed.shape[-1]
    prior = self.img_step(prev_state, prev_action, None, sample)
    if self._shared:
      post = self.img_step(prev_state, prev_action, embed, sample)
    else:
      if self._temp_post:
        x = tf.concat([prior['deter'], embed], -1)
      else:
        x = embed
      for i in range(self._layers_output):
        x = self.get(f'obi{i}', tfkl.Dense, self._hidden, self._act)(x)
      stats = self._suff_stats_layer('obs', x)
      if sample:
        stoch = self.get_dist(stats).sample()
      else:
        stoch = self.get_dist(stats).mode()
      post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, embed=None, sample=True):
    prev_stoch = prev_state['stoch']
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    if self._shared:
      if embed is None:
        shape = prev_action.shape[:-1] + [self._embed]
        embed = tf.zeros(shape, prev_action.dtype)
      x = tf.concat([prev_stoch, prev_action, embed], -1)
    else:
      x = tf.concat([prev_stoch, prev_action], -1)
    for i in range(self._layers_input):
      x = self.get(f'ini{i}', tfkl.Dense, self._hidden, self._act)(x)
    for _ in range(self._rec_depth):
      deter = prev_state['deter']
      x, deter = self._cell(x, [deter])
      deter = deter[0]  # Keras wraps the state in a list.
    for i in range(self._layers_output):
      x = self.get(f'imo{i}', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('ims', x)
    if sample:
      stoch = self.get_dist(stats).sample()
    else:
      stoch = self.get_dist(stats).mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      mean = {
          'none': lambda: mean,
          'tanh5': lambda: 5.0 * tf.math.tanh(mean / 5.0),
      }[self._mean_act]()
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'abs': lambda: tf.math.abs(std + 1),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, scale):
    kld = tfd.kl_divergence
    dist = lambda x: self.get_dist(x, tf.float32)
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(dist(lhs), dist(rhs))
      loss = tf.reduce_mean(tf.maximum(value, free))
    else:
      value_lhs = value = kld(dist(lhs), dist(sg(rhs)))
      value_rhs = kld(dist(sg(lhs)), dist(rhs))
      loss_lhs = tf.maximum(tf.reduce_mean(value_lhs), free)
      loss_rhs = tf.maximum(tf.reduce_mean(value_rhs), free)
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    loss *= scale
    return loss, value


class ConvEncoder(tools.Module):

  def __init__(
      self, depth=32, act=tf.nn.relu, kernels=(4, 4, 4, 4)):
    self._act = act
    self._depth = depth
    self._kernels = kernels

  def __call__(self, obs):
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** i * self._depth
      x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
    x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
    # print('Encoder output:', x.shape)
    shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
    return tf.reshape(x, shape)


class ConvDecoder(tools.Module):

  def __init__(
      self, depth=32, act=tf.nn.relu, shape=(64, 64, 3), kernels=(5, 5, 6, 6),
      thin=True):
    self._act = act
    self._depth = depth
    self._shape = shape
    self._kernels = kernels
    self._thin = thin

  def __call__(self, features, dtype=None):
    ConvT = tfkl.Conv2DTranspose
    if self._thin:
      x = self.get('hin', tfkl.Dense, 32 * self._depth, None)(features)
      x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    else:
      x = self.get('hin', tfkl.Dense, 128 * self._depth, None)(features)
      x = tf.reshape(x, [-1, 2, 2, 32 * self._depth])
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 1) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        depth = self._shape[-1]
        act = None
      x = self.get(f'h{i}', ConvT, depth, kernel, 2, activation=act)(x)
    # print('Decoder output:', x.shape)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    if dtype:
      mean = tf.cast(mean, dtype)
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseHead(tools.Module):

  def __init__(
      self, shape, layers, units, act=tf.nn.elu, dist='normal', std=1.0):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._act = act
    self._dist = dist
    self._std = std

  def __call__(self, features, dtype=None):
    x = features
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    mean = self.get(f'hmean', tfkl.Dense, np.prod(self._shape))(x)
    mean = tf.reshape(mean, tf.concat(
        [tf.shape(features)[:-1], self._shape], 0))
    if self._std == 'learned':
      std = self.get(f'hstd', tfkl.Dense, np.prod(self._shape))(x)
      std = tf.nn.softplus(std) + 0.01
      std = tf.reshape(std, tf.concat(
          [tf.shape(features)[:-1], self._shape], 0))
    else:
      std = self._std
    if dtype:
      mean, std = tf.cast(mean, dtype), tf.cast(std, dtype)
    if self._dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, std), len(self._shape))
    if self._dist == 'huber':
      return tfd.Independent(
          tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape))
    if self._dist == 'binary':
      return tfd.Independent(tfd.Bernoulli(mean), len(self._shape))
    raise NotImplementedError(self._dist)


class ActionHead(tools.Module):

  def __init__(
      self, size, layers, units, act=tf.nn.elu, dist='trunc_normal',
      init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
    # assert min_std <= 2
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._action_disc = action_disc
    self._temp = temp() if callable(temp) else temp
    self._outscale = outscale

  def __call__(self, features, dtype=None):
    x = features
    for index in range(self._layers):
      kw = {}
      if index == self._layers - 1 and self._outscale:
        kw['kernel_initializer'] = tf.keras.initializers.VarianceScaling(
            self._outscale)
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act, **kw)(x)
    if self._dist == 'tanh_normal':
      # https://www.desmos.com/calculator/rcmcf5jwe7
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      mean = tf.tanh(mean)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'tanh_normal_5':
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      mean = 5 * tf.tanh(mean / 5)
      std = tf.nn.softplus(std + 5) + 5
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'normal':
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'normal_1':
      mean = self.get(f'hout', tfkl.Dense, self._size)(x)
      if dtype:
        mean = tf.cast(mean, dtype)
      dist = tfd.Normal(mean, 1)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'trunc_normal':
      # https://www.desmos.com/calculator/mmuvuhnyxo
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      x = tf.cast(x, tf.float32)
      mean, std = tf.split(x, 2, -1)
      mean = tf.tanh(mean)
      std = 2 * tf.nn.sigmoid(std / 2) + self._min_std
      dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'onehot':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      x = tf.cast(x, tf.float32)
      dist = tools.OneHotDist(x, dtype=dtype)
      dist = tools.DtypeDist(dist, dtype)
    elif self._dist == 'onehot_gumble':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      temp = self._temp
      dist = tools.GumbleDist(temp, x, dtype=dtype)
    else:
      raise NotImplementedError(self._dist)
    return dist


class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]

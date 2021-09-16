import collections
import logging
import os
import pathlib
import re
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf

import agent
import common

from common import Config
from common import GymWrapper
from common import Flags
from common import ResizeImage
from common import NormalizeAction
from common import OneHotAction
from common import TimeLimit
from common import TerminalOutput
from common import JSONLOutput
from common import TensorBoardOutput

configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))
configs = common.Config({k: defaults.update(v) for k, v in configs.items()})


def train(env, config, outputs=None):

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  outputs = outputs or [
      common.TerminalOutput(),
      common.JSONLOutput(config.logdir),
      common.TensorBoardOutput(config.logdir),
  ]
  replay = common.Replay(logdir / 'train_episodes', **config.replay)
  step = common.Counter(replay.stats['total_steps'])
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video = common.Every(config.log_every)
  should_expl = common.Until(config.expl_until)

  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'Episode has {length} steps and return {score:.1f}.')
    logger.scalar('return', score)
    logger.scalar('length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{key}', ep[key].max(0).mean())
    if should_video(step):
      for key in config.log_keys_video:
        logger.video(f'policy_{key}', ep[key])
    logger.add(replay.stats)
    logger.write()

  print('Create envs.')
  act_space = env.act_space['action']
  num_actions = act_space.n if config.discrete else act_space.shape[-1]
  driver = common.Driver([env])
  driver.on_episode(per_episode)
  driver.on_step(lambda tran, worker: step.increment())
  driver.on_step(replay.add_step)
  driver.on_reset(replay.add_step)

  prefill = max(0, config.prefill - replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(num_actions, config.discrete)
    driver(random_agent, steps=prefill, episodes=1)
    driver.reset()

  print('Create agent.')
  dataset = iter(replay.dataset(**config.dataset))
  shapes = {k: v.shape[2:] for k, v in dataset.element_spec.items()}
  agnt = agent.Agent(config, logger, step, shapes)
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(dataset))
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(dataset))
  policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')

  def train_step(tran, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        mets = train_agent(next(dataset))
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(dataset)))
      logger.write(fps=True)
  driver.on_step(train_step)

  while step < config.steps:
    logger.write()
    driver(policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')

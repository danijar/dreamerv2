import collections
import functools
import logging
import os
import pathlib
import sys
import warnings

import ruamel.yaml as yaml

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))

configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())

import numpy as np
import tensorflow as tf

import agent
import common


def main():

  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)

  def make_env(mode):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
      return common.DMC(task, config.action_repeat, config.image_size)
    elif suite == 'atari':
      return common.Atari(
          task, config.action_repeat, config.image_size, config.grayscale,
          life_done=False, sticky_actions=True, all_actions=True)
    elif suite == 'crafter':
      return common.Crafter(
          outdir=mode == 'train' and pathlib.Path(config.logdir) / 'crafter')
    else:
      raise NotImplementedError(suite)

  run(make_env, config)


def run(make_env, config):

  def wrapped_make_env(mode):
    env = make_env(mode)
    env = common.DictSpaces(env)
    if hasattr(env.action_space['action'], 'n'):
      env = common.OneHotAction(env)
    else:
      env = common.NormalizeAction(env)
    env = common.TimeLimit(env, config.time_limit)
    env = common.RewardObs(env)
    env = common.ResetObs(env)
    return env

  logdir = pathlib.Path(config.logdir).expanduser()
  config = config.update(
      steps=config.steps // config.action_repeat,
      eval_every=config.eval_every // config.action_repeat,
      log_every=config.log_every // config.action_repeat,
      time_limit=config.time_limit // config.action_repeat,
      prefill=config.prefill // config.action_repeat)

  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  print('Logdir', logdir)
  train_replay = common.Replay(logdir / 'train_replay', config.replay_size)
  eval_replay = common.Replay(logdir / 'eval_replay', config.time_limit or 1)
  step = common.Counter(train_replay.total_steps)
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)
  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)

  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    replay_ = dict(train=train_replay, eval=eval_replay)[mode]
    replay_.add(ep)
    logger.scalar(f'{mode}_transitions', replay_.num_transitions)
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    logger.scalar(f'{mode}_eps', replay_.num_episodes)
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      logger.video(f'{mode}_policy', ep['image'])
    logger.write()

  print('Create envs.')
  train_envs = [wrapped_make_env('train') for _ in range(config.num_envs)]
  eval_envs = [wrapped_make_env('eval') for _ in range(config.num_envs)]
  action_space = train_envs[0].action_space['action']
  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda _: step.increment())
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))

  prefill = max(0, config.prefill - train_replay.total_steps)
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(action_space)
    train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()

  print('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, logger, action_space, step, train_dataset)
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    config.pretrain and print('Pretrain agent.')
    for _ in range(config.pretrain):
      agnt.train(next(train_dataset))

  def train_step(tran):
    if should_train(step):
      for _ in range(config.train_steps):
        _, mets = agnt.train(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(train_dataset)), prefix='train')
      logger.write(fps=True)
  train_driver.on_step(train_step)

  while step < config.steps:
    logger.write()
    print('Start evaluation.')
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    eval_policy = functools.partial(agnt.policy, mode='eval')
    eval_driver(eval_policy, episodes=config.eval_eps)
    print('Start training.')
    train_driver(agnt.policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  main()

import gym
import gym_minigrid
import dreamerv2.api as dv2

config = dv2.defaults.update({
    'logdir': '~/logdir/minigrid',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()

env = gym.make('MiniGrid-DoorKey-6x6-v0')
env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
dv2.train(env, config)

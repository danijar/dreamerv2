import gym
import gym_minigrid
import dreamerv2.api as dv2

config = dv2.configs.crafter.update({
    'logdir': '~/logdir/minigrid',
    'discrete': True,
    'log_every': 1e3,
    'train_every': 10,
    'actor.dist': 'onehot',
    'actor_grad': 'reinforce',
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()

env = gym.make('MiniGrid-DoorKey-6x6-v0')
env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
env = dv2.GymWrapper(env)
env = dv2.ResizeImage(env, (64, 64))
env = dv2.OneHotAction(env)

dv2.train(env, config)

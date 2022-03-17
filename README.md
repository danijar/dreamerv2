**Status:** Stable release

[![PyPI](https://img.shields.io/pypi/v/dreamerv2.svg)](https://pypi.python.org/pypi/dreamerv2/#history)

# Mastering Atari with Discrete World Models

Implementation of the [DreamerV2][website] agent in TensorFlow 2. Training
curves for all 55 games are included.

<p align="center">
<img width="90%" src="https://imgur.com/gO1rvEn.gif">
</p>

If you find this code useful, please reference in your paper:

```
@article{hafner2020dreamerv2,
  title={Mastering Atari with Discrete World Models},
  author={Hafner, Danijar and Lillicrap, Timothy and Norouzi, Mohammad and Ba, Jimmy},
  journal={arXiv preprint arXiv:2010.02193},
  year={2020}
}
```

[website]: https://danijar.com/dreamerv2

## Method

DreamerV2 is the first world model agent that achieves human-level performance
on the Atari benchmark. DreamerV2 also outperforms the final performance of the
top model-free agents Rainbow and IQN using the same amount of experience and
computation. The implementation in this repository alternates between training
the world model, training the policy, and collecting experience and runs on a
single GPU.

![World Model Learning](https://imgur.com/GRC9QAw.png)

DreamerV2 learns a model of the environment directly from high-dimensional
input images. For this, it predicts ahead using compact learned states. The
states consist of a deterministic part and several categorical variables that
are sampled. The prior for these categoricals is learned through a KL loss. The
world model is learned end-to-end via straight-through gradients, meaning that
the gradient of the density is set to the gradient of the sample.

![Actor Critic Learning](https://imgur.com/wH9kJ2O.png)

DreamerV2 learns actor and critic networks from imagined trajectories of latent
states. The trajectories start at encoded states of previously encountered
sequences. The world model then predicts ahead using the selected actions and
its learned state prior. The critic is trained using temporal difference
learning and the actor is trained to maximize the value function via reinforce
and straight-through gradients.

For more information:

- [Google AI Blog post](https://ai.googleblog.com/2021/02/mastering-atari-with-discrete-world.html)
- [Project website](https://danijar.com/dreamerv2/)
- [Research paper](https://arxiv.org/pdf/2010.02193.pdf)

## Using the Package

The easiest way to run DreamerV2 on new environments is to install the package
via `pip3 install dreamerv2`. The code automatically detects whether the
environment uses discrete or continuous actions. Here is a usage example that
trains DreamerV2 on the MiniGrid environment:

```python
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
```

## Manual Instructions

To modify the DreamerV2 agent, clone the repository and follow the instructions
below. There is also a Dockerfile available, in case you do not want to install
the dependencies on your system.

Get dependencies:

```sh
pip3 install tensorflow==2.6.0 tensorflow_probability ruamel.yaml 'gym[atari]' dm_control
```

Train on Atari:

```sh
python3 dreamerv2/train.py --logdir ~/logdir/atari_pong/dreamerv2/1 \
  --configs atari --task atari_pong
```

Train on DM Control:

```sh
python3 dreamerv2/train.py --logdir ~/logdir/dmc_walker_walk/dreamerv2/1 \
  --configs dmc_vision --task dmc_walker_walk
```

Monitor results:

```sh
tensorboard --logdir ~/logdir
```

Generate plots:

```sh
python3 common/plot.py --indir ~/logdir --outdir ~/plots \
  --xaxis step --yaxis eval_return --bins 1e6
```

## Docker Instructions

The [Dockerfile](https://github.com/danijar/dreamerv2/blob/main/Dockerfile)
lets you run DreamerV2 without installing its dependencies in your system. This
requires you to have Docker with GPU access set up.

Check your setup:

```sh
docker run -it --rm --gpus all tensorflow/tensorflow:2.4.2-gpu nvidia-smi
```

Train on Atari:

```sh
docker build -t dreamerv2 .
docker run -it --rm --gpus all -v ~/logdir:/logdir dreamerv2 \
  python3 dreamerv2/train.py --logdir /logdir/atari_pong/dreamerv2/1 \
    --configs atari --task atari_pong
```

Train on DM Control:

```sh
docker build -t dreamerv2 . --build-arg MUJOCO_KEY="$(cat ~/.mujoco/mjkey.txt)"
docker run -it --rm --gpus all -v ~/logdir:/logdir dreamerv2 \
  python3 dreamerv2/train.py --logdir /logdir/dmc_walker_walk/dreamerv2/1 \
    --configs dmc_vision --task dmc_walker_walk
```

## Tips

- **Efficient debugging.** You can use the `debug` config as in `--configs
atari debug`. This reduces the batch size, increases the evaluation
frequency, and disables `tf.function` graph compilation for easy line-by-line
debugging.

- **Infinite gradient norms.** This is normal and described under loss scaling in
the [mixed precision][mixed] guide. You can disable mixed precision by passing
`--precision 32` to the training script. Mixed precision is faster but can in
principle cause numerical instabilities.

- **Accessing logged metrics.** The metrics are stored in both TensorBoard and
JSON lines format. You can directly load them using `pandas.read_json()`. The
plotting script also stores the binned and aggregated metrics of multiple runs
into a single JSON file for easy manual plotting.

[mixed]: https://www.tensorflow.org/guide/mixed_precision

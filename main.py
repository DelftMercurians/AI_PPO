import time

import jax
from jax import lax
from jax import random as jr
from jax import numpy as jnp
from jax import tree_util as jtu

import matplotlib.pyplot as plt
from IPython.display import clear_output
from brax import envs
from src.wrappers import ActionTanhConstraintWrapper, ObservationNormalizingWrapper, Agent
from src.dataclasses import HyperParameters
from src.ppo import train

xdata, ydata = [], []


def progress(step_num, metrics):
    xdata.append(step_num)
    ydata.append(metrics["eval/episode_reward"])
    clear_output(wait=True)

    plt.xlim([0, 40_000_000])
    plt.ylim([0, 4000])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")

    plt.plot(xdata, ydata, "b")
    plt.show()


t0 = time.time()

env = envs.create(env_name="ant", backend="spring")

agent = Agent(jr.PRNGKey(42), env.observation_size, env.action_size)
agent = ObservationNormalizingWrapper(agent)
agent = ActionTanhConstraintWrapper(agent, range_low=-1.0, range_high=1.0)

agent, metrics = train(
    agent,
    params=HyperParameters(
        env=env,
        num_timesteps=40_000_000,
        num_evals=10,
        reward_scaling=0.1,
        episode_length=1000,
        unroll_length=5,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        batch_size=1024,
        eval_batch_size=256,
        gae_lambda=0.97,
        max_gradient_norm=1.0,
        clipping_epsilon=0.3,
    ),
    progress=progress,
)

print(f"Training (and tracing) done in {time.time() - t0:.2f} seconds")
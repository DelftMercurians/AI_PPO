import jax
from jax import lax
from jax import random as jr
from jax import numpy as jnp
from jax import tree_util as jtu

from jaxtyping import Array, Float, Int32, PRNGKeyArray

import equinox as eqx

from .dataclasses import Action, LogNormalDistribution


class Critic(eqx.Module):
    """MLP, that outputs the TD residual for the given state."""

    structure: list

    def __init__(self, key: PRNGKeyArray, observation_size: int):
        output_size = 1  # output is the value, TD residual, always a single output

        key1, key2, key3, key4 = jr.split(key, 4)
        self.structure = [
            eqx.nn.Linear(observation_size, 64, key=key1),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=key2),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=key3),
            jax.nn.tanh,
            eqx.nn.Linear(64, output_size, key=key4),
        ]

    def __call__(self, x):
        for operator in self.structure:
            x = operator(x)
        return x


class MeanNetwork(eqx.Module):
    """MLP, that outputs a mean for the Action distribution, given state."""

    structure: list

    def __init__(self, key: PRNGKeyArray, observation_size: int, action_size: int):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.structure = [
            eqx.nn.Linear(observation_size, 64, key=key1),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=key2),
            jax.nn.tanh,
            eqx.nn.Linear(64, 64, key=key3),
            jax.nn.tanh,
            eqx.nn.Linear(64, action_size, key=key4),
        ]

        # scaling down the weights of the output layer improves performance
        self.structure = eqx.tree_at(
            where=lambda s: s[-1].weight,
            pytree=self.structure,
            replace_fn=lambda weight: weight * 0.01,
        )

    def __call__(self, x):
        for operator in self.structure:
            x = operator(x)
        return x


class Actor(eqx.Module):
    """A module, that outputs action distribution for a particular state."""

    mean_network: MeanNetwork
    log_std: jax.Array  # Trainable array

    def __init__(
        self,
        key: PRNGKeyArray,
        observation_size: int,
        action_size: int,
        initial_std: float,
    ):
        self.mean_network = MeanNetwork(key, observation_size, action_size)
        self.log_std = jnp.ones((action_size,)) * jnp.log(initial_std)

    def __call__(self, x):
        return LogNormalDistribution(self.mean_network(x), self.log_std)


class ActorCritic(eqx.Module):
    """Allows to use functionality of both Actor and Critic via the same entity."""

    obs_size: int
    act_size: int
    critic: Critic
    actor: Actor

    def __init__(
        self,
        key: PRNGKeyArray,
        observation_size: int,
        action_size: int,
        initial_actor_std: float = 0.5,
    ):
        self.obs_size = observation_size
        self.act_size = action_size

        key1, key2 = jax.random.split(key, 2)
        self.critic = Critic(key1, observation_size)
        self.actor = Actor(key2, observation_size, action_size, initial_actor_std)

    def get_value(self, observation):
        return self.critic(observation)

    def get_action(self, key: PRNGKeyArray, observation):
        distr = self.actor(observation)
        action = distr.sample(key)
        return Action(raw=action, transformed=action, distr=distr)

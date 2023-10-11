from beartype.typing import Callable, Union, Any

import jax
from jax import random as jr
from jax import numpy as jnp

from jaxtyping import Array, Float, Int32, PRNGKeyArray

import optax

import equinox as eqx

from brax import envs

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wrappers import BaseWrapper, ActionTanhConstraintWrapper


class LogNormalDistribution(eqx.Module):
    """Multivariate Log Normal distribution with diagonal covariance"""

    mean: Float[Array, "*batch mean_dim"]
    log_std: Float[Array, "*batch mean_dim"]

    def get_pdf(self, value: Float[Array, "*batch mean_dim"]) -> Float[Array, "*batch"]:
        value = eqx.error_if(
            value,
            value.shape != self.mean.shape,
            "Wrong shapes for the mean/value of action distr",
        )
        value = eqx.error_if(
            value,
            value.shape != self.log_std.shape,
            "Wrong shapes for the std/value of action distr",
        )

        normalized = (value - self.mean) / jnp.exp(self.log_std)
        return jax.scipy.stats.norm.logpdf(normalized).sum()

    def sample(self, key: PRNGKeyArray):
        return jr.normal(key, self.mean.shape) * jnp.exp(self.log_std) + self.mean

    def entropy(self):
        return self.log_std.sum() * 0.5  # entropy without constant factor


class Action(eqx.Module):
    """
    Action class represents a single action taken by an agent.
    Additionally stores some useful data.

    raw: action that was a direct output of an Actor-Critic model
    transformed: action that was applied on the environment
    distr: distribution from which raw action was sampled
    """

    raw: Float[Array, "action_size"] = None
    transformed: Float[Array, "action_size"] = None
    distr: LogNormalDistribution = None

    def postprocess(self, apply: Callable):
        return Action(
            raw=self.raw, transformed=apply(self.transformed), distr=self.distr
        )


class ValueRange(eqx.Module):
    low: float
    high: float


class HyperParameters(eqx.Module):
    """All the parameters for the algorithm you will ever need."""

    # parameters that are forcefully static, and you are not allowed to change them,
    # unless you really want to retrace most of the jitted functions
    env: envs.Env = eqx.field(static=True)
    episode_length: int = eqx.field(static=True)

    num_timesteps: int = eqx.field(default=30_000_000, static=True)
    seed: int = eqx.field(default=0, static=True)
    num_evals: int = eqx.field(default=10, static=True)

    # parameters that are 'changeable' throughout training.
    # Does not mean that you should change them :)
    learning_rate: float = 1e-4
    clipping_epsilon: float = 0.2
    batch_size: int = 32
    eval_batch_size: int = 16
    entropy_cost: float = 0
    discounting: float = 0.99
    gae_lambda: float = 0.95
    num_updates_per_batch: int = 2
    num_minibatches: int = 16
    unroll_length: int = 10
    reward_scaling: float = 1.0
    max_gradient_norm: float = 0.5
    value_loss_factor: float = 0.25


class Transition(eqx.Module):
    """Represents a transition between two adjacent environment states."""

    observation: Float[Array, "*batch obs_size"]  # observation on the current state
    action: Action  # action that was taken on the current state
    reward: Float[Array, "*batch"]  # reward, that was given as the result of the action
    next_observation: Float[Array, "*batch obs_size"]  # next observation
    extras: dict  # any simulator-extracted hints, like end of episode signal


class Optimizer(eqx.Module):
    """An optax optimizer wrapped with its state together."""

    optimizer: optax.GradientTransformation = eqx.field(static=True)
    state: optax.OptState

    def update(self, grads):
        out_updates, new_state = self.optimizer.update(grads, self.state)
        return out_updates, Optimizer(self.optimizer, new_state)


class Environment(eqx.Module):
    """A Brax environment, wrapped with its state and step counter together."""

    env: envs.base.Env = eqx.field(static=True)
    state: envs.base.State
    # todo: fix with the new version of jaxtyping
    steps_done: Union[int, Int32[Array, ""]] = eqx.field(default=0, converter=jnp.asarray)


class TrainingState(eqx.Module):
    optimizer: Optimizer
    # todo: i get beartype.roar.BeartypeCallHintForwardRefException:
    #  Forward reference "ppo_brax_equinox.dataclasses.ActionTanhConstraintWrapper" unimportable.
    #  if i specify the class
    agent: Any  # "BaseWrapper", or specifically "ActionTanhConstraintWrapper"
    env: Environment

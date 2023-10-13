import jax
from jax import lax
from jax import random as jr
from jax import numpy as jnp
from jax import tree_util as jtu

from jaxtyping import Array, Float, Int32, PRNGKeyArray
from beartype.typing import Tuple

import equinox as eqx

from .models import ActorCritic
from .dataclasses import ValueRange, Action


class BaseWrapper(eqx.Module):
    """
    Parent class of any wrapper.

    next: the wrapper (or anything else) that is considered the next one.
    params: persistent properties (parameters) of this particular wrapper.
    """

    next: "BaseWrapper" = None
    params: eqx.Module = None

    def get_trainable(self):
        # returns a PyTree with the same structure as self,
        # but every leaf except children of Actor-Critic are replaced by None
        return eqx.filter(
            self,
            filter_spec=lambda x: isinstance(x, ActorCritic),
            is_leaf=lambda x: isinstance(x, ActorCritic),
        )

    def get_obs_size(self):
        return self.next.get_obs_size()

    def get_act_size(self):
        return self.next.get_act_size()

    def set_next(self, new_next):
        return eqx.tree_at(lambda wrapper: wrapper.next, self, new_next)

    def get_action(self, key: PRNGKeyArray, observation):
        out, new_next = self.next.get_action(key, observation)
        return out, self.set_next(new_next)

    def get_value(self, observation):
        out, new_next = self.next.get_value(observation)
        return out, self.set_next(new_next)

    def config(self, **kwargs):
        return self.set_next(self.next.config(**kwargs))


class Agent(BaseWrapper):
    """
    Transparent, unmutable wrapper.
    Used so that even pure Actor-Critic can be used as a wrapper.
    """

    def __init__(self, *args, **kwargs):
        self.next: ActorCritic = ActorCritic(*args, **kwargs)

    def set_next(self, _):
        # immutable -> don't change the self.next
        return self

    def get_action(self, key: PRNGKeyArray, observation: Float[Array, "*batch obs_size"]) -> Tuple[
            Action, "Agent"]:
        return self.next.get_action(key, observation), self

    def get_value(self, observation: Float[Array, "*batch obs_size"]) -> Tuple[
            Float[Array, "*batch size"], "Agent"]:
        return self.next.get_value(observation), self

    def config(self, **kwargs):
        return self

    def get_obs_size(self):
        return self.next.obs_size

    def get_act_size(self):
        return self.next.act_size


class _RunningStats(eqx.Module):
    """Stores/updates the parameters of the running distribution."""

    mean: jax.Array
    M2: jax.Array  # sum of second moments of the samples (sum of variances)
    n: Int32[Array, "1"]
    size: int  # the length of a single observation (obs_size)

    # we are initializing n with two so that we don't get division by zero, ever
    # this biases the running statistics, but not really that much
    def __init__(self, size: int, mean: Float[Array, "*batch size"] = None, M2: Float[Array, "*batch size"] = None,
                 n=jnp.int32(2)):
        self.size = size
        self.mean = (jnp.zeros(size)) if mean is None else mean
        self.M2 = (jnp.zeros(size) + 1e-6) if M2 is None else M2
        self.n = n

    def process(self, obs: Float[Array, "*batch size"]) -> Tuple[Float[Array, "*batch size"], "_RunningStats"]:
        std = jnp.sqrt(self.M2 / self.n)
        std = eqx.error_if(
            std,
            std.shape != obs.shape,
            "Standard deviation should have the same shape as the observation, "
            + f"std shape is {std.shape} but observation shape is {obs.shape}",
        )

        # clip std, so that we don't get extreme values
        std = jnp.clip(std, 1e-6, 1e6)

        # clip the extreme outliers -> more stability during training.
        # by Chebyshev inequality, ~99% of values are not clipped.
        processed = jnp.clip((obs - self.mean) / std, -10, 10)

        return lax.stop_gradient(processed), self.update_single(obs)

    def update_single(self, obs) -> "_RunningStats":
        return self.update(obs[None, :])

    def update(self, obs: Float[Array, "*batch size"]) -> "_RunningStats":
        obs = eqx.error_if(
            obs,
            len(obs.shape) != 2 or obs.shape[1] != self.size,
            f"Batched observation should have the shape of (_, {self.size}),"
            + f"but got {obs.shape}",
        )

        n = self.n + obs.shape[0]

        diff_to_old_mean = obs - self.mean
        new_mean = self.mean + diff_to_old_mean.sum(axis=0) / n

        diff_to_new_mean = obs - new_mean
        var_upd = jnp.sum(diff_to_old_mean * diff_to_new_mean, axis=0)
        M2 = self.M2 + var_upd

        return _RunningStats(self.size, mean=new_mean, M2=M2, n=n)


class ObservationNormalizingWrapper(BaseWrapper):
    """Wrapper, that normalizes the observations during 'runtime'."""

    def __init__(self, next: BaseWrapper, params=None):
        self.next = next
        self.params: _RunningStats = (
            _RunningStats(self.next.get_obs_size()) if params is None else params
        )

    def get_value(self, observation: Float[Array, "*batch obs_size"]) -> Tuple[
            Float[Array, "*batch size"], "ObservationNormalizingWrapper"]:
        observation, updated_params = self.params.process(observation)
        out, new_next = self.next.get_value(observation)
        return out, ObservationNormalizingWrapper(new_next, updated_params)

    def get_action(self, key: PRNGKeyArray, observation: Float[Array, "*batch obs_size"]) -> Tuple[
            Action, "ObservationNormalizingWrapper"]:
        observation, updated_params = self.params.process(observation)
        out, new_next = self.next.get_action(key, observation)
        return out, ObservationNormalizingWrapper(new_next, updated_params)

    def config(self, **kwargs) -> "ObservationNormalizingWrapper":
        params = self.params

        if "force_running_stats_update" in kwargs:
            params = self.params.update(kwargs.get("force_running_stats_update"))

        return ObservationNormalizingWrapper(self.next.config(**kwargs), params)


class ActionTanhConstraintWrapper(BaseWrapper):
    """'Softly' constraints an action to the provided range."""

    def __init__(self, next, range_low=-1.0, range_high=1.0):
        self.next = next
        self.params: ValueRange = ValueRange(range_low, range_high)

    def get_action(self, key: PRNGKeyArray, observation: Float[Array, "*batch obs_size"]) -> Tuple[
            Action, "ActionTanhConstraintWrapper"]:
        action, new_next = self.next.get_action(key, observation)

        scale = self.params.high - self.params.low
        offset = self.params.low
        normalizing_function = lambda x: (jnp.tanh(x) / 2.0 + 0.5) * scale + offset
        action = action.postprocess(normalizing_function)

        return action, self.set_next(new_next)


class ActionExactConstraintWrapper(BaseWrapper):
    """Clips an action to the given range."""

    def __init__(self, next, range_low, range_high):
        self.next = next
        self.params: ValueRange = ValueRange(range_low, range_high)
        self.range_low = range_low
        self.range_high = range_high

    def get_action(self, key: PRNGKeyArray, observation: Float[Array, "*batch obs_size"]) -> Tuple[
            Action, "ActionExactConstraintWrapper"]:
        action, new_next = self.next.get_action(key, observation)
        action = action.postprocess(
            lambda x: jnp.clip(x, self.range_low, self.range_high)
        )
        return action, self.set_next(new_next)

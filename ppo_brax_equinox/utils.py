import jax
from brax import envs
from jax import lax
from jax import random as jr
from jax import numpy as jnp
from jax import tree_util as jtu

from jaxtyping import Array, Float, Int32, PRNGKeyArray
from beartype.typing import Tuple

import equinox as eqx

from collections.abc import Callable

from .dataclasses import Transition


# Two utility functions that are similar to other Equinox filtered wrappers.
#
# In the implementation we are constantly using lax.scan, and hence wrapping it,
# so that it can consume partially-static PyTrees seems nice.

@eqx.filter_jit
def filter_scan(f: Callable, init, xs, *args, **kwargs):
    """Same as lax.scan, but allows to have eqx.Module in carry"""
    init_dynamic_carry, static_carry = eqx.partition(init, eqx.is_array)

    def to_scan(dynamic_carry, x):
        carry = eqx.combine(dynamic_carry, static_carry)
        new_carry, out = f(carry, x)
        dynamic_new_carry, _ = eqx.partition(new_carry, eqx.is_array)
        return dynamic_new_carry, out

    out_carry, out_ys = lax.scan(to_scan, init_dynamic_carry, xs, *args, **kwargs)
    return eqx.combine(out_carry, static_carry), out_ys


@eqx.filter_jit
def filter_cond(pred, true_f: Callable, false_f: Callable, *args):
    """Same as lax.cond, but allows to return eqx.Module"""
    dynamic_true, static_true = eqx.partition(true_f(*args), eqx.is_array)
    dynamic_false, static_false = eqx.partition(false_f(*args), eqx.is_array)

    static_part = eqx.error_if(
        static_true,
        static_true != static_false,
        "Filtered conditional arguments should have the same static part",
    )

    dynamic_part = lax.cond(pred, lambda *_: dynamic_true, lambda *_: dynamic_false)
    return eqx.combine(dynamic_part, static_part)


# The next two functions are needed to collect trajectories from the environment,
# provided "policy": a function that given an observation,
# returns the **Action** PyTree.
def actor_step(key: PRNGKeyArray, env, env_state, policy: Callable, extra_fields):
    """Makes a single step with the provided policy in the environment."""
    keys_policy = jr.split(key, env_state.obs.shape[0])
    action, _ = eqx.filter_vmap(policy)(keys_policy, env_state.obs)
    next_state = env.step(env_state, action.transformed)

    return next_state, Transition(
        observation=env_state.obs,
        action=action,
        reward=next_state.reward,
        next_observation=next_state.obs,
        # extract requested additional fields
        extras={x: next_state.info[x] for x in extra_fields},
    )


def generate_unroll(
    key: PRNGKeyArray, env, env_state, policy: Callable, unroll_length, extra_fields
) -> Tuple[envs.base.State, Transition]:
    """Collects trajectories of given unroll length."""

    def f(carry, _):
        current_key, state = carry
        current_key, next_key = jr.split(current_key)

        next_state, transition = actor_step(
            current_key, env, state, policy, extra_fields=extra_fields
        )
        return (next_key, next_state), transition

    (_, final_state), data = filter_scan(f, (key, env_state), (), length=unroll_length)
    return final_state, data

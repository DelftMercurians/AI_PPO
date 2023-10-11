import time

import jax
from jax import random as jr
from jax import numpy as jnp

from jaxtyping import Array, Float, Int32, PRNGKeyArray

import equinox as eqx

from brax import envs

from .utils import generate_unroll


# The **Evaluator** evaluates a given agent on the environment,
# providing quite a bit of interesting information about the results.
class Evaluator:
    """
    Evaluates agent on the environment.
    It is not jittable, since run_evaluation needs time.time()
    """

    def __init__(self, eval_env, agent, num_eval_envs, episode_length):
        self._eval_walltime = 0.0
        self.eval_env = envs.training.EvalWrapper(eval_env)
        self.episode_length = episode_length
        self.num_eval_envs = num_eval_envs
        self._steps_per_unroll = episode_length * num_eval_envs

    @eqx.filter_jit
    def evaluate(self, key: PRNGKeyArray, agent):
        reset_keys = jr.split(key, self.num_eval_envs)
        eval_first_state = self.eval_env.reset(reset_keys)
        return generate_unroll(
            key,
            self.eval_env,
            eval_first_state,
            agent.get_action,
            unroll_length=self.episode_length,
            extra_fields=("truncation",),
        )[0]

    def run_evaluation(
        self, key: PRNGKeyArray, agent, training_metrics, aggregate_episodes: bool = True
    ):
        t = time.time()
        eval_state = self.evaluate(key, agent)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [jnp.mean, jnp.std]:
            suffix = "_std" if fn == jnp.std else ""
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(value) if aggregate_episodes else value
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics["eval/avg_episode_length"] = jnp.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics

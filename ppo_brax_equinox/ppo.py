import jax
from jax import lax
from jax import random as jr
from jax import numpy as jnp
from jax import tree_util as jtu

import equinox as eqx

import optax

from brax import envs

from .utils import filter_scan, filter_cond, generate_unroll
from .dataclasses import TrainingState, Environment, Optimizer
from .evaluator import Evaluator


@eqx.filter_jit
def compute_gae(
    truncation, rewards, values, bootstrap_value, gae_lambda, time_discount
):
    """
    Computes Generalized Advantage Estimation (GAE).
    https://arxiv.org/abs/1506.02438 (formula 16)
    """

    def to_scan(gae_t_plus_one, inputs):
        delta_t, truncation_t = inputs
        gae_t = gae_t_plus_one * time_discount * gae_lambda + delta_t
        gae_t = gae_t * (1 - truncation_t)
        return gae_t, gae_t

    next_values = jnp.concatenate([values[1:], bootstrap_value], axis=0)
    deltas = rewards + time_discount * next_values - values
    _, advantages = lax.scan(to_scan, 0.0, (deltas, truncation), reverse=True)
    return advantages + values, advantages


@eqx.filter_jit
def compute_loss(key: jr.PRNGKey, data, agent, params):
    """
    Computes standard PPO loss on a single trajectory, with clipped surrogate objective.
    https://arxiv.org/abs/1707.06347 (link to the PPO paper)
    """
    key_actions = jr.split(key, data.observation.shape[0])

    # the second returned value is updated agent (stacked unroll_length times)
    new_actions, _ = eqx.filter_vmap(agent.get_action)(key_actions, data.observation)
    baseline, _ = eqx.filter_vmap(agent.get_value)(data.observation)

    baseline = eqx.error_if(
        baseline,
        baseline.shape != (params.unroll_length, 1),
        f"Baseline Values should have shape {(params.unroll_length, 1)}, "
        + f"but got {baseline.shape}",
    )

    # (unroll_length, 1) -> (unroll_length,)
    baseline = baseline.reshape((params.unroll_length,))
    bootstrap_value, _ = agent.get_value(data.next_observation[-1])

    rewards = data.reward * params.reward_scaling
    behaviour_actions = data.action

    get_log_pdf_from_distr = lambda action, distr: distr.get_pdf(action)

    # compute log of pdfs of the the old actions for new/old distributions
    # we use "raw" actions, since they are directly sampled from the distributions
    new_distr_log_pdf = eqx.filter_vmap(get_log_pdf_from_distr)(
        behaviour_actions.raw, new_actions.distr
    )
    old_distr_log_pdf = eqx.filter_vmap(get_log_pdf_from_distr)(
        behaviour_actions.raw, behaviour_actions.distr
    )
    # ratio of probabilities that the old (behavioural) action will be taken
    rho = jnp.exp(new_distr_log_pdf - old_distr_log_pdf)

    target_values, advantages = compute_gae(
        truncation=data.extras["truncation"],
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        gae_lambda=params.gae_lambda,
        time_discount=params.discounting,
    )

    # stop gradients for numerical stability, and since they are "meaningless"
    # the point is that we train with respect to these parameters,
    # and allowing them too to be trainable too leads to weird results
    target_values = lax.stop_gradient(target_values)
    advantages = lax.stop_gradient(advantages)

    # compute clipped policy loss
    surrogate_loss1 = rho * advantages
    surrogate_loss2 = (
        jnp.clip(rho, 1 - params.clipping_epsilon, 1 + params.clipping_epsilon)
        * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # compute value loss
    v_error = target_values - baseline
    v_loss = jnp.mean(v_error * v_error) * params.value_loss_factor

    # and finally, the entropy loss (we encourage higher entropy)
    entropy = jnp.mean(new_actions.distr.entropy())
    entropy_loss = params.entropy_cost * -entropy

    # sum all the losses up
    total_loss = policy_loss + v_loss + entropy_loss
    return total_loss, {
        "total_loss": total_loss,
        "rho": rho,
        "p_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
    }


@eqx.filter_jit
def clip_by_norm(x, max_norm=1.0):
    """Clips the norm of the vector, with some whistles"""
    max_norm = eqx.error_if(max_norm, max_norm < 0, "Clip norm should be non-negative")
    norm = lax.cond(
        jnp.array_equal(x, jnp.zeros_like(x)),
        lambda *_: jnp.float32(1.0),
        lambda *_: jnp.linalg.norm(x),
    )
    return x * jnp.minimum(max_norm / norm, 1.0)


@eqx.filter_jit
def sgd_step(key: jr.PRNGKey, optimizer, agent, data, params):
    """
    Does not do a single SGD step. It does a bunch of optimizer steps on the minibatches.

    Most importantly, there are tree things that this function does:
        1. The passed data is shuffled, and partitioned into minibatches.
        2. The mean loss over each of the minibatches is computed.
        3. Using reverse autodiff (**eqx.filter_value_and_grad**),
                the gradients are computed, and we update the agent.
    """
    key_perm, key_grad = jr.split(key)

    def convert_data(x: jax.Array):
        """Shuffles input data, and partitions in into the minibatches"""
        x = jr.permutation(key_perm, x)
        x = jnp.reshape(x, (params.num_minibatches, -1) + x.shape[1:])
        return x

    shuffled_data = jtu.tree_map(convert_data, data)

    # check that the shape is correct
    desired_shape = (params.num_minibatches, params.batch_size, params.unroll_length)
    shuffled_data = eqx.error_if(
        shuffled_data,
        shuffled_data.observation.shape[:-1] != desired_shape,
        f"Minibatch data shape is wrong, should be {desired_shape} "
        + f"but was {shuffled_data.observation.shape[:-1]}",
    )

    def minibatch_step_to_scan(carry, data):
        key, optimizer, agent = carry
        key_next, key_loss = jr.split(key)

        def batched_loss(agent, data):
            loss_f = lambda data, agent: compute_loss(key_loss, data, agent, params)
            loss_value, metrics = eqx.filter_vmap(loss_f, in_axes=(0, None))(
                data, agent
            )
            # we waste less memory by computing mean for everything in metrics
            return loss_value.mean(), jtu.tree_map(lambda x: x.mean(axis=0), metrics)

        get_value_and_grad = eqx.filter_value_and_grad(
            jtu.Partial(batched_loss, data=data), has_aux=True
        )
        (loss, metrics), grads = get_value_and_grad(agent)

        # gradient clipping -> more stable training
        def filter_and_clip_grads(grad, trainable):
            # making sure that we update only trainable stuff
            # since grads are also computed for all the wrappers params
            # which breaks optax
            if grad is None or trainable is None:
                return None
            return clip_by_norm(grad, params.max_gradient_norm)

        grads = jtu.tree_map(filter_and_clip_grads, grads, agent.get_trainable())

        updates, new_optimizer = optimizer.update(grads)
        new_agent = eqx.apply_updates(agent, updates)

        return (key_next, new_optimizer, new_agent), metrics

    (_, new_optimizer, new_agent), metrics = filter_scan(
        minibatch_step_to_scan,
        (key_grad, optimizer, agent),
        shuffled_data,
        length=params.num_minibatches,
    )

    return new_optimizer, new_agent, jtu.tree_map(lambda x: x.mean(axis=0), metrics)


@eqx.filter_jit
def training_step(carry, _, params):
    """
    The function consists of three meaningful parts:
        1. Collection of **batch_size** number of trajectories of **unroll_length** length
            from the provided environment, with the current policy.
            The agent remains constant throughout the collection of the trajectory.
        2. Split of the collected data into a convenient shape,
            which is (a lot, **unroll_length**),
            so that later we can vmap or map over the zero-th axis,
            and the mapped function will get just a single trajectory.
        3. Update of the agent. We do an optimizer step on the computed PPO loss,
            and then we update the observation normalizing wrapper with just collected trajectories data.
            The running statistics is updated in a batch since it is easier, and it does not hurt performance much.
    :param carry:
    :param _:
    :param params:
    :return:
    """
    key, training_state = carry
    key_sgd, key_generate_unroll, key_next = jr.split(key, 3)

    agent = training_state.agent
    env = training_state.env

    # wrap the functions, so that lax.scan can use them
    def unroll_to_scan(carry, _):
        key_unroll, state, policy = carry
        key_unroll, next_key = jr.split(key_unroll)

        next_state, generated = generate_unroll(
            key_unroll,
            env.env,
            state,
            policy,
            params.unroll_length,
            extra_fields=("truncation",)
            # truncation is the signal that environment evaluation has finished
            # in this case we 'reset' the advantage computation
        )
        return (next_key, next_state, policy), generated

    def sgd_step_to_scan(carry, _):
        key_sgd, optimizer, agent = carry
        key_sgd, next_key = jr.split(key_sgd)

        new_opt, new_ag, metrics = sgd_step(key_sgd, optimizer, agent, data, params)
        return (next_key, new_opt, new_ag), metrics

    # generate unroll with the current policy
    (_, new_env_state, *_), data = filter_scan(
        unroll_to_scan,
        (key_generate_unroll, env.state, agent.get_action),
        (),
        length=params.num_minibatches,
    )

    # transform all the data from the unroll into more convenient shape
    data = jtu.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jtu.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)

    # check the correctness of the shape
    target_data_shape = (
        params.batch_size * params.num_minibatches,
        params.unroll_length,
        params.env.observation_size,
    )
    data = eqx.error_if(
        data,
        data.observation.shape != target_data_shape,
        f"Reshaped unroll data (observation) must have shape of "
        f"{target_data_shape} but had {data.observation.shape}",
    )

    # optimize the model, do a few optimizer steps
    (_, new_optimizer, new_agent), metrics = filter_scan(
        sgd_step_to_scan,
        (key_sgd, training_state.optimizer, agent),
        (),
        length=params.num_updates_per_batch,
    )

    # update the normalizing wrapper with collected observations
    reshaped_data = data.observation.reshape(-1, data.observation.shape[-1])
    new_agent = new_agent.config(force_running_stats_update=reshaped_data)

    # the first training iteration used only to update observation normalizing wrapper
    # otherwise the first step would be random
    get_new = lambda *_: (new_optimizer, new_agent)
    get_old = lambda *_: (training_state.optimizer, training_state.agent)
    new_optimizer, new_agent = filter_cond(env.steps_done != 0, get_new, get_old)

    env_steps_made = params.batch_size * params.num_minibatches * params.unroll_length

    # construct new training state, with all the updated stuff
    new_training_state = TrainingState(
        optimizer=new_optimizer,
        agent=new_agent,
        env=Environment(env.env, new_env_state, env.steps_done + env_steps_made),
    )
    return (key_next, new_training_state), metrics


@eqx.filter_jit
def training_epoch(key: jr.PRNGKey, training_state, params):
    """
    Not only runs a bunch of **training_steps**,
    but it also figures out the number of **training_steps** to run
    judging by the required number of timesteps that we want to train for, and other parameters.
    :param key:
    :param training_state:
    :param params:
    :return:
    """
    env_step_per_training_step = (
        params.batch_size * params.unroll_length * params.num_minibatches
    )
    num_evals_after_init = params.num_evals - 1 if params.num_evals > 1 else 1
    num_training_steps_per_epoch = 1 + params.num_timesteps // (
        num_evals_after_init * env_step_per_training_step
    )

    (_, training_state), metrics = filter_scan(
        jtu.Partial(training_step, params=params),
        (key, training_state),
        (),
        length=num_training_steps_per_epoch,
    )

    return training_state, metrics


def train(agent, params, progress=lambda *_: None):
    """
    Initializes some brax-wrapped environments, some variables, some states,
    and then just run **training_epoch** a few times.
    **train** is not JIT-traced, so that user-defined **progress** can do anything:
    printing something, logging to wandb, plotting graphs, etc.
    :param agent:
    :param params:
    :param progress:
    :return:
    """
    key = jr.PRNGKey(params.seed)
    key_local, key_env = jr.split(key, 2)

    env = envs.training.wrap(params.env, episode_length=params.episode_length)
    reset_fn = jax.jit(env.reset)

    # we are using Adam, since, well, everybody uses Adam
    optimizer = optax.adam(learning_rate=params.learning_rate)
    trainable_agent_arrays = eqx.filter(agent.get_trainable(), eqx.is_array)
    key_envs = jr.split(key_env, params.batch_size)

    training_state = TrainingState(
        optimizer=Optimizer(optimizer, optimizer.init(trainable_agent_arrays)),
        agent=agent,
        env=Environment(env, reset_fn(key_envs), steps_done=0),
    )

    evaluator = Evaluator(
        env,
        agent,
        num_eval_envs=params.eval_batch_size,
        episode_length=params.episode_length,
    )

    # run very first eval before any training
    metrics = {}
    if params.num_evals > 1:
        metrics = evaluator.run_evaluation(
            key, training_state.agent, training_metrics={}
        )
        progress(0, metrics)

    num_of_epochs = max(params.num_evals - 1, 1)
    for it in range(num_of_epochs):
        # update the keys
        key_epoch, key_local, key_eval = jr.split(key_local, 3)

        # train
        training_state, training_metrics = training_epoch(
            key_epoch, training_state, params
        )

        # reset the environment state
        key_envs = jr.split(key_local, params.batch_size)
        training_state = eqx.tree_at(
            where=lambda t: t.env.state,
            pytree=training_state,
            replace=reset_fn(key_envs),
        )

        # update metrics
        metrics = evaluator.run_evaluation(
            key_eval, training_state.agent, training_metrics
        )
        progress(training_state.env.steps_done, metrics)

    return (training_state.agent, metrics)

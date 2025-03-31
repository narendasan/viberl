import marimo

__generated_with = "0.11.7"
app = marimo.App(width="full")


@app.cell
def _():
    import logging
    import os
    import time

    import jax
    import marimo as mo
    from brax import envs
    from brax.io import html
    from rejax import PPO

    from viberl.env import render_gymnax
    from viberl.utils import (
        argparser,
        build_eval_callback,
        create_checkpointer_from_config,
        create_eval_logger,
        create_mlflow_logger,
        generate_experiment_config,
        load_ckpt,
        setup_logger,
    )

    return (
        PPO,
        argparser,
        build_eval_callback,
        create_checkpointer_from_config,
        create_eval_logger,
        create_mlflow_logger,
        envs,
        generate_experiment_config,
        html,
        jax,
        load_ckpt,
        logging,
        mo,
        os,
        render_gymnax,
        setup_logger,
        time,
    )


@app.cell
def _(time):
    config = {
        "experiment": {
            "root_seed": 42,
            "num_agent_seeds": 16,
            "ckpt_dir": "ckpts",
            "tags": ["test"],
            "algorithm": "ppo",
            "max_ckpt_to_keep": 5,
            "results_dir": "results",
            "log_dir": "logs",
            "experiment_name": f"gymnax_experiment1-{time.time()}-seed42-steps1000000000-lr0.0003-test",
        },
        "algorithm": {
            "env": "gymnax/Breakout-MinAtar",
            "total_timesteps": 10000000,
            "eval_freq": 100000,
            "num_envs": 256,
            "num_steps": 128,
            "num_epochs": 16,
            "num_minibatches": 16,
            "learning_rate": 0.0003,
            "max_grad_norm": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "agent_kwargs": {"activation": "relu"},
        },
    }
    return (config,)


@app.cell
def _(jax):
    print(jax.default_backend())
    print(jax.local_devices())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""JAX handles reproducablity through the key system. Starting from a root key (can be thought of as a seed) keys can be split to control PRNG across a vector of agents. Here we create $N$ splits of the root key, one for each agent we will train Under the hood, each PPO instance will also split their key $M$ times for each of the envs it will train across"""
    )
    return


@app.cell
def _(config, jax):
    root_key = jax.random.key(config["experiment"]["root_seed"])
    agent_keys = jax.random.split(root_key, config["experiment"]["num_agent_seeds"])
    return agent_keys, root_key


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Here we create a vector of $N$ agents that we will train seeded with their own key derived from the root key"""
    )
    return


@app.cell
def _(PPO, config):
    algo = PPO.create(**config["algorithm"])
    return (algo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        We then insert the callbacks for logging and reporting on training process into each agent
        These transforms are functional so you get a new agent out instead of modifying in place
        """
    )
    return


@app.cell
def _(
    algo,
    build_eval_callback,
    config,
    create_checkpointer_from_config,
    create_eval_logger,
    create_mlflow_logger,
):
    algo_w_callback = algo.replace(
        eval_callback=build_eval_callback(
            algo,
            [
                create_eval_logger(),
                create_mlflow_logger(config),
                create_checkpointer_from_config(config),
            ],
        )
    )
    return (algo_w_callback,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """We then can vectorize across $N \times M$ instances of agents and envs and train these in parallel"""
    )
    return


@app.cell
def _(agent_keys, algo, jax):
    vmap_train = jax.jit(jax.vmap(algo.train)).lower(agent_keys).compile()
    return (vmap_train,)


@app.cell
def _(agent_keys, vmap_train):
    train_states, results = vmap_train(agent_keys)
    print(results)
    return results, train_states


@app.cell
def _(algo_w_callback, jax, train_states):
    from viberl.utils import tree_unstack

    train_state = tree_unstack(train_states)[0]

    policy_fn = algo_w_callback.make_act(train_states)
    jit_policy_fn = jax.jit(policy_fn)
    return jit_policy_fn, policy_fn, train_state, tree_unstack


@app.cell
def _(config, jit_policy_fn, render_gymnax, root_key):
    vis, reward = render_gymnax(jit_policy_fn, config, root_key)
    return reward, vis


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

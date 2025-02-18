import marimo

__generated_with = "0.11.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import logging
    from functools import partial

    import jax
    from rejax import PPO

    import wandb
    from rl_sandbox.env._trajectories import collect_trajectories
    from rl_sandbox.env._visualize import collect_rollouts
    from rl_sandbox.utils import (argparser, build_eval_callback,
                                  create_checkpointer_from_config, create_eval_logger,
                                  create_wandb_logger, generate_experiment_config, load_ckpt)

    from brax import envs
    from brax.io import html
    from IPython.display import HTML
    return (
        HTML,
        PPO,
        argparser,
        build_eval_callback,
        collect_rollouts,
        collect_trajectories,
        create_checkpointer_from_config,
        create_eval_logger,
        create_wandb_logger,
        envs,
        generate_experiment_config,
        html,
        jax,
        load_ckpt,
        logging,
        mo,
        partial,
        wandb,
    )


@app.cell
def _(jax):
    print(jax.default_backend())
    print(jax.local_devices())
    return


@app.cell
def _(generate_experiment_config, logging, wandb):
    logging.basicConfig(level=logging.DEBUG)
    config = generate_experiment_config("config/test.toml")

    wandb.init(
        project="rl-sandbox",
        group=config["experiment"]["experiment_name"],
        tags=config["experiment"]["tags"],
        config=config
    )
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""JAX handles reproducablity through the key system. Starting from a root key (can be thought of as a seed) keys can be split to control PRNG across a vector of agents. Here we create $N$ splits of the root key, one for each agent we will train Under the hood, each PPO instance will also split their key $M$ times for each of the envs it will train across""")
    return


@app.cell
def _(config, jax):
    root_key = jax.random.key(config["experiment"]["root_seed"])
    agent_keys = jax.random.split(root_key, config["experiment"]["num_agent_seeds"])
    return agent_keys, root_key


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here we create a vector of $N$ agents that we will train seeded with their own key derived from the root key""")
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
    create_wandb_logger,
):
    algo_w_callback = algo.replace(eval_callback=build_eval_callback(algo, [
        create_eval_logger(),
        create_wandb_logger(),
        create_checkpointer_from_config(config)
    ]))
    return (algo_w_callback,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We then can vectorize across $N \times M$ instances of agents and envs and train these in parallel""")
    return


@app.cell
def _(agent_keys, algo_w_callback, jax):
    vmap_train = jax.jit(jax.vmap(algo_w_callback.train))
    train_states, results = vmap_train(agent_keys)
    return results, train_states, vmap_train


@app.cell
def _(collect_trajectories, config, partial):
    vmap_collect_trajectories = partial(collect_trajectories, env_name=config["algorithm"]["env"], env_config=config["algorithm"]["env_params"], num_envs=config["algorithm"]["num_envs"], max_steps_in_episode=1000)
    return (vmap_collect_trajectories,)


@app.cell
def _():
    #t_results, trajectories = jax.vmap(vmap_collect_trajectories)(agents, train_states, agent_keys)
    return


@app.cell
def _(a, jax, train_states):
    train_states.seed[a]
    import numpy as np
    jax.random.key_data(train_states.seed[0])
    return (np,)


@app.cell
def _():
    # root_key_, rollout_key = jax.random.split(root_key)
    # rollouts = {}
    # for a in range(config["experiment"]["num_agent_seeds"]):
    #     algo_ = PPO.create(**config["algorithm"])
    #     train_state = load_ckpt(algo_, config["experiment"]["ckpt_dir"], config["experiment"]["experiment_name"], key=train_states.seed[a], tag="best")
    #     rollout_key, env_key = jax.random.split(rollout_key)
    #     rollout = collect_rollouts(algo, train_state, env_key, env_name=config["algorithm"]["env"], env_config=config["algorithm"]["env_params"], num_envs=config["algorithm"]["num_envs"], max_steps_in_episode=1000)
    #     rollouts[a] = rollout
    return


@app.cell
def _(rollouts):
    rollouts[0][2][1]
    return


@app.cell
def _():
    # env = envs.create(env_name="ant", backend="positional")

    # states = []
    # for i in range(2):
    #     print(i)
    #     states.append(rollouts[0][2][1][i].state.pipeline_state)

    # HTML(html.render(env, states))
    return


@app.cell
def _(PPO, config, envs, html, jax, load_ckpt, train_states):
    #@title Visualizing a trajectory of the learned inference function
    import jax.numpy as jnp
    # create an env with auto-reset
    env = envs.create(env_name="ant", backend="positional")

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    algo0 = PPO.create(**config["algorithm"])
    train_state = load_ckpt(algo0, config["experiment"]["ckpt_dir"], config["experiment"]["experiment_name"], key=train_states.seed[0], tag="best")
    inference_fn = algo0.make_act(train_state)
    jit_inference_fn = jax.jit(inference_fn)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    reward = 0
    for _ in range(10000):
      reward += state.reward
      rollout.append(state.pipeline_state)
      act_rng, rng = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_rng)
      state = jit_env_step(state, act)

    html.save("/home/naren/Downloads/test.html", env.sys.tree_replace({'opt.timestep': env.dt}), rollout)
    print(f"Saved: {reward}")
    return (
        act,
        act_rng,
        algo0,
        env,
        inference_fn,
        jit_env_reset,
        jit_env_step,
        jit_inference_fn,
        jnp,
        reward,
        rng,
        rollout,
        state,
        train_state,
    )


@app.cell
def _(env, html, mo, rollout):
    mo.Html(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

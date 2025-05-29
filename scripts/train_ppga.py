import logging

import flax
import jax
from rejax.compat import create

from viberl.algorithms.ppga import State as PPGAState
from viberl.algorithms.ppga import Config as PPGAConfig
from viberl.algorithms.ppga import caclulate_dcd_gradients, move_mean_agent
from viberl.utils import (
    argparser,
    generate_experiment_config,
    setup_logger,
)
from viberl.utils import build_eval_callback, create_eval_logger, create_checkpointer_from_config, create_wandb_logger

parser = argparser()
args = parser.parse_args()
config = generate_experiment_config(args.config_file)

setup_logger(config)
_LOGGER = logging.getLogger(__name__)

# wandb.init(
#     project="rl-sandbox",
#     group=config["experiment"]["experiment_name"],
#     tags=config["experiment"]["tags"],
#     config=config
# )
# JAX handles reproducablity through the key system. Starting from a root key (can be thought of as a seed)
# keys can be split to control PRNG across a vector of agents
# Here we create N splits of the root key, one for each agent we will train
# Under the hood, each PPO instance will also split their key M times for each of the envs it will train across
root_key = jax.random.key(config["experiment"]["root_seed"])
agent_keys = jax.random.split(root_key, config["experiment"]["num_agent_seeds"])

# Here we create a vector of N agents that we will train seeded with their own key derived from the root key
env_info = create(config["experiment"]["env_name"])
cfg = PPGAConfig.from_dict(config["rl"])
rngs = flax.nnx.Rngs(root_key)
ppga_state = PPGAState.from_env(cfg, env_info=env_info, rngs=rngs)
print(ppga_state)
params = ppga_state[0].actor_ts.params
flattened_params, restore_fn = jax.flatten_util.ravel_pytree(params)

for i in range(config["qd"]["num_iterations"]):
    solution_batch = scheduler.ask_dqd()
    state.load_solution(solution_batch)
# We then insert the callbacks for logging and reporting on training process into each agent
# These transforms are functional so you get a new agent out instead of modifying in place
# algo = algo.replace(
#     eval_callback=build_eval_callback(
#         algo,
#         [
#             create_eval_logger(),
#             create_mlflow_logger(config),
#             create_checkpointer_from_config(config),
#         ],
#     )
# )
#


# We then can vectorize across NxM instances of agents and envs and train these in parallel
# This can just be run as JIT, but further gains can be gotten from lowering and AOT compiling the training function
_LOGGER.info("Compiling training function...")
#vmap_train = jax.jit(jax.vmap(train)).lower(agent_keys).compile()

# _LOGGER.info("Training...")
train(
    ppo_state,
    cfg,
    env_info,
    root_key,
    eval_callback=build_eval_callback([
        create_eval_logger(),
        create_checkpointer_from_config(config),
        create_wandb_logger(config),
    ])
)
# print(results)
# agent_ts = tree_unstack(train_states)

# policy = algo.make_act(agent_ts[0])
# jit_policy = jax.jit(policy)
# vis, reward = render_gymnax(jit_policy, config, root_key)
# vis.animate(f"{os.getcwd()}/{config['experiment']['results_dir']}/train_ppo.gif")

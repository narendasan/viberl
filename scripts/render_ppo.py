import os
import logging

from pathlib import Path
import flax
import jax
from brax.io import html

from rejax.compat import create

from viberl.algorithms.ppo import Config, State
from viberl.utils import (
    generate_experiment_config,
    setup_logger,
)

from viberl.env import render_brax
from viberl.utils import (
    argparser_for_eval,
    load_ckpt,
)

parser = argparser_for_eval()
args = parser.parse_args()

config = generate_experiment_config(Path(args.experiment_path) / "config.toml")
root_key = jax.random.key(config["experiment"]["root_seed"])

setup_logger(config)
_LOGGER = logging.getLogger(__name__)

env_info = create("brax/walker2d")
cfg = Config.from_dict(config["algorithm"])
rngs = flax.nnx.Rngs(root_key)
ppo_state = State.from_env(cfg, env_info=env_info, rngs=rngs)

ppo_state_dict = load_ckpt(
    ppo_state,
    args.experiment_path,
    run_name=args.seed_name,
    step=args.step,
)
ppo_state.load_ckpt_dict(ppo_state_dict)
print(ppo_state)

os.makedirs(f"{os.getcwd()}/{config['experiment']['results_dir']}", exist_ok=True)

policy_fn = ppo_state.policy_fn()
jit_policy_fn = jax.jit(policy_fn)

reward = 0.0
rollout, env, reward = render_brax(jit_policy_fn, "brax/walker2d", root_key)
html.save(
    f"{os.getcwd()}/{config['experiment']['results_dir']}/{args.seed_name}_{args.step}_{config['experiment']['experiment_name']}.html",
    env.sys.tree_replace({"opt.timestep": env.dt}),
    rollout,
)

_LOGGER.info(f"Saved [{args.seed_name} (step: {args.step})]: Reward: {reward}")

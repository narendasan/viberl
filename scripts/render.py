import logging
import os

from brax.io import html
import jax
from rejax import PPO

from viberl.utils import (
    argparser_for_eval,
    generate_experiment_config,
    load_ckpt,
    setup_logger,
)
from viberl.env import (render_brax, render_gymnax)

parser = argparser_for_eval()
args = parser.parse_args()

config = generate_experiment_config(args.config_file)
root_key = jax.random.key(config["experiment"]["root_seed"])

setup_logger(config)
_LOGGER = logging.getLogger(__name__)

algo0 = PPO.create(**config["algorithm"])

train_state = load_ckpt(
    algo0,
    config["experiment"]["ckpt_dir"],
    args.experiment,
    run_name=args.seed_name,
    step=args.step,
    rng=root_key)

os.makedirs(f"{os.getcwd()}/{config['experiment']['results_dir']}", exist_ok=True)

policy_fn = algo0.make_act(train_state)
jit_policy_fn = jax.jit(policy_fn)

reward = 0
if config["algorithm"]["env"].startswith("brax"):
    rollout, env, reward = render_brax(jit_policy_fn, config, root_key)
    html.save(f"{os.getcwd()}/{config['experiment']['results_dir']}/{args.seed_name}_{args.step}_{args.experiment}.html", env.sys.tree_replace({'opt.timestep': env.dt}), rollout)
elif config["algorithm"]["env"].startswith("gymnax"):
    vis, reward = render_gymnax(jit_policy_fn, config, root_key)
    vis.animate(f"{os.getcwd()}/{config['experiment']['results_dir']}/{args.seed_name}_{args.step}_{args.experiment}.gif")
else:
    raise ValueError(f"Unsupported environment: {config['algorithm']['env']}")

_LOGGER.info(f"Saved [{args.seed_name} (step: {args.step})]: Reward: {reward}")

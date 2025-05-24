import argparse
import datetime
import tomllib
from pathlib import Path
from typing import Any, Dict


def _generate_experiment_name(config: Dict[str, Any]) -> str:
    prefix = config["experiment"]["name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    prefix += f"_{timestamp}"
    seed = config["experiment"]["root_seed"]
    prefix += f"_seed{seed}"
    steps = config["algorithm"]["total_timesteps"]
    prefix += f"_steps{steps:.2E}"
    #lr = config["algorithm"]["actor_lr"]
    #prefix += f"_alr{lr:.2E}"
    #lr = config["algorithm"]["critic_lr"]
    #prefix += f"_clr{lr:.2E}"

    if "tags" in config["experiment"]:
        tags = config["experiment"]["tags"]
        tag_slugs = ""
        for t in tags:
            tag_slugs += f"-{t}"
        return f"{prefix}{tag_slugs}"
    else:
        return prefix


def generate_experiment_config(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        config = tomllib.load(f)
    config["experiment"]["experiment_name"] = _generate_experiment_name(config)
    return config


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    return parser


def argparser_for_eval() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-name", type=str)
    parser.add_argument("--experiment-path", type=str)
    parser.add_argument("--step", type=str, default="best")
    return parser

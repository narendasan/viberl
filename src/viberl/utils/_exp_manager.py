import argparse
import datetime
import tomllib
from pathlib import Path
from typing import Any, Dict


def _generate_experiment_name(config: Dict[str, Any]) -> str:
    prefix = config["experiment"]["name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    prefix += f"-{timestamp}"
    seed = config["experiment"]["root_seed"]
    prefix += f"-seed{seed}"
    steps = config["algorithm"]["total_timesteps"]
    prefix += f"-steps{steps}"
    lr = config["algorithm"]["learning_rate"]
    prefix += f"-lr{lr}"

    tags = config["experiment"]["tags"]
    if tags is not None or len(tags) != 0:
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
    parser = argparser()
    parser.add_argument("--seed-name", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--step", type=str, default="best")
    return parser

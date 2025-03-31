import logging
import os
from typing import Any, Dict


def setup_logger(config: Dict[str, Any]) -> None:
    os.makedirs(f"{os.getcwd()}/{config['experiment']['log_dir']}", exist_ok=True)
    lvl = (
        logging.getLevelName(config["experiment"]["log_level"])
        if "log_level" in config["experiment"]
        else logging.INFO
    )
    logging.basicConfig(
        level=lvl,
        handlers=[
            logging.FileHandler(
                f"{config['experiment']['log_dir']}/{config['experiment']['experiment_name']}.log"
            ),
            logging.StreamHandler(),
        ],
        force=True,  # Needed since orbax does some weird stuff
    )
    logging.getLogger("jax._src.xla_bridge").propagate = False
    logging.getLogger("absl").propagate = False
    return

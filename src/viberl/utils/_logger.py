import os

import logging
import logging.config
from typing import Any, Dict
import absl.logging

absl.logging.set_verbosity(absl.logging.WARNING)

def setup_logger(config: Dict[str, Any]) -> None:
    os.makedirs(f"{os.getcwd()}/{config['experiment']['log_dir']}", exist_ok=True)
    lvl = (
        logging.getLevelName(config["experiment"]["log_level"])
        if "log_level" in config["experiment"]
        else logging.INFO
    )
    logging.config.dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "brief": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "file": {
                    "level": lvl,
                    "class": "logging.FileHandler",
                    "filename": f"{config['experiment']['log_dir']}/{config['experiment']['experiment_name']}.log",
                    "formatter": "standard",
                },
                "console": {
                    "level": lvl,
                    "class": "logging.StreamHandler",
                    "formatter": "brief",
                },
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["file", "console"],
                    "level": lvl,
                    "propagate": True,
                },
            },
            "force": True,  # Needed since orbax does some weird stuff
        })

    logging.getLogger("jax._src.xla_bridge").propagate = False
    logging.getLogger("jax._src").propagate = False
    logging.getLogger("absl").propagate = False
    return

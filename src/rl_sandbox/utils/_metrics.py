import copy
import wandb
import logging

_LOGGER = logging.getLogger(__name__)

def _wandb_runner(config: Dict[str,Any]):
    wandb.init(project="rl_sandbox", config=config)

    while True:
        with open
    wandb.watch(model)
    wandb.finish()

class WandBWatchDog:
    def __init__(self, config: Dict[str,Any]):
        self.config = copy.deepcopy(config)

    def watch(self, jax.Array):

        self.wandb.watch(model)


    def close(self):
        self.wandb.finish()

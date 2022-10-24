from datetime import datetime

import hydra
from omegaconf import DictConfig
from transformers import set_seed
from transformers.utils.logging import get_logger

from data import DataModule
from utils import (
    prepare_trackers,
    load_trainer,
    save_results,
    cleanup_run,
)

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    set_seed(cfg.training_args.seed)
    dm = DataModule(cfg)

    cfg.run_start = datetime.utcnow().strftime("%Y-%d-%m_%H-%M-%S")

    prepare_trackers(cfg)

    for fold in range(cfg.folds_to_run):

        cfg.fold = fold

        trainer = load_trainer(cfg, dm)

        trainer.train()

        save_results(trainer, cfg)

        cleanup_run(trainer)

if __name__ == "__main__":
    main()

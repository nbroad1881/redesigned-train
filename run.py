from pathlib import Path
from functools import partial
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from transformers.utils.logging import get_logger

from data import DataModule
from utils import set_wandb_env_vars, compute_metrics


logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    dm = DataModule(cfg)

    t_args = TrainingArguments(**(dict(cfg.training_args)))

    set_seed(t_args.seed)

    if "wandb" in t_args.report_to:
        set_wandb_env_vars(cfg)

    with t_args.main_process_first():
        dm.prepare_dataset()

    model_config = AutoConfig.from_pretrained(
        cfg.model.model_name_or_path,
        problem_type=cfg.model.problem_type,
        num_labels=len(dm.label2id),
        label2id=dm.label2id,
        id2label=dm.id2label,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        attention_dropout_prob=cfg.attention_dropout_prob,
    )

    run_start = datetime.utcnow().strftime("%Y-%d-%m_%H-%M-%S")

    for fold in range(cfg.data.kfolds):

        t_args.output_dir = f"{Path.cwd()/(run_start+'_f'+str(fold))}"

        train_ds = dm.get_train_dataset(fold)
        eval_ds = dm.get_eval_dataset(fold)

        model = AutoModelForSequenceClassification(
            cfg.model.model_name_or_path, config=model_config
        )

        collator = DataCollatorWithPadding(
                dm.tokenizer, pad_to_multiple_of=cfg.data.pad_multiple
            )

        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=dm.tokenizer,
            data_collator=collator,
            compute_metrics=partial(compute_metrics, label2id=dm.label2id),
        )

        trainer.train()


if __name__ == "__main__":
    main()

import gc
from pathlib import Path
from functools import partial
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from transformers.integrations import WandbCallback
from transformers.utils.logging import get_logger

from data import DataModule
from utils import set_wandb_env_vars, compute_metrics, NewWandbCB, MaskAugmentationTrainer, SaveCallback
from modeling.base import BaseModel

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    dm = DataModule(cfg)

    t_args = TrainingArguments(**(dict(cfg.training_args)))

    set_seed(t_args.seed)

    USING_WANDB =  "wandb" in t_args.report_to
    if USING_WANDB:
        import wandb
        print("wandb")
        set_wandb_env_vars(cfg)

    with t_args.main_process_first():
        dm.prepare_dataset()

    run_start = datetime.utcnow().strftime("%Y-%d-%m_%H-%M-%S")
    cfg.run_start = run_start

    for fold in range(cfg.folds_to_run):
        
        cfg.fold = fold

        t_args.output_dir = f"{Path.cwd()/(run_start+'_f'+str(fold))}"
        
        model_config = AutoConfig.from_pretrained(
            cfg.model.model_name_or_path,
            problem_type=cfg.data.problem_type,
            num_labels=len(dm.label2id),
            label2id=dm.label2id,
            id2label=dm.id2label,
            hidden_dropout_prob=cfg.model.hidden_dropout_prob,
            attention_probs_dropout_prob=cfg.model.attention_probs_dropout_prob,
        )
        model_config.update({
            "multisample_dropout": OmegaConf.to_container(cfg.model.multisample_dropout),
            "output_layer_norm": cfg.model.output_layer_norm,
            "classifier_dropout_prob": cfg.model.classifier_dropout_prob,
            "loss_fn": cfg.model.loss_fn,
        })

        train_ds = dm.get_train_dataset(fold)
        eval_ds = dm.get_eval_dataset(fold)

        model = BaseModel.from_pretrained(
            cfg.model.model_name_or_path, config=model_config
        )

        collator = DataCollatorWithPadding(
                dm.tokenizer, pad_to_multiple_of=cfg.data.pad_multiple
            )
        
        ### Set up callbacks ###
    
        callbacks = []
        if USING_WANDB:
            callbacks.append(NewWandbCB(cfg))

        callbacks.append(SaveCallback(
            threshold_score=cfg.threshold_score, 
            metric_name=t_args.metric_for_best_model,
            greater_is_better=t_args.greater_is_better,
            weights_only=True,
        ))
        
        if cfg.data.mask_augmentation:
            trainer_class = partial(MaskAugmentationTrainer, cfg=cfg)
        else:
            trainer_class = Trainer
            
        metrics = partial(compute_metrics, data_module=dm, data_cfg=cfg.data)
        

        trainer = trainer_class(
            model=model,
            args=t_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=dm.tokenizer,
            data_collator=collator,
            compute_metrics=metrics,
            callbacks=callbacks,
        )
        
        trainer.remove_callback(WandbCallback)
        
        if USING_WANDB:
            wandb.init(config=OmegaConf.to_container(cfg))

        trainer.train()

        best_metric_score = getattr(
            trainer.model.config,
            f"best_{t_args.metric_for_best_model}", 
            cfg.threshold_score,
        )
        trainer.log({f"best_{t_args.metric_for_best_model}": best_metric_score})

        run_id = wandb.run.id if USING_WANDB else ""
        run_name = wandb.run.name if USING_WANDB else ""
        
        
        model.config.update({"wandb_id": run_id, "wandb_name": run_name})
        model.config.save_pretrained(t_args.output_dir)

        if t_args.push_to_hub:
            push_to_hub(
                trainer,
                config=cfg,
                metrics={f"best_{t_args.metric_for_best_model}": best_metric_score},
                wandb_run_id=run_id,
            )

        if USING_WANDB:
            wandb.finish()

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        


if __name__ == "__main__":
    main()

import os
from collections.abc import Mapping
from typing import Any, Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch
import torch

import torch
import numpy as np
from transformers import TrainingArguments
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.utils import logging
from transformers.file_utils import is_torch_tpu_available

logger = logging.get_logger(__name__)


def set_wandb_env_vars(cfg):
    """
    Set environment variables from the config dict object.
    The environment variables can be picked up by wandb in Trainer.
    """

    os.environ["WANDB_ENTITY"] = getattr(cfg.wandb, "entity", "")
    os.environ["WANDB_PROJECT"] = getattr(cfg.wandb, "project", "")
    os.environ["WANDB_RUN_GROUP"] = getattr(cfg.wandb, "group", "")
    os.environ["WANDB_JOB_TYPE"] = getattr(cfg.wandb, "job_type", "")
    os.environ["WANDB_NOTES"] = getattr(cfg.wandb, "notes", "")
    if cfg.wandb.tags:
        os.environ["WANDB_TAGS"] = ",".join(cfg.wandb.tags)


class NewWandbCB(WandbCallback):
    def __init__(self, run_config):
        super().__init__()
        self.run_config = run_config

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.
        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/integrations/huggingface). You can also override the following environment
        variables:
        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training. Use along with
                *TrainingArguments.load_best_model_at_end* to upload best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to disable gradient logging or `"all"` to
                log gradients and parameters.
            WANDB_PROJECT (`str`, *optional*, defaults to `"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (`bool`, *optional*, defaults to `False`):
                Whether or not to disable wandb entirely. Set *WANDB_DISABLED=true* to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict(), **self.run_config}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            run_name = os.getenv("WANDB_NAME")

            if self._wandb.run is None:
                tags = os.getenv("WANDB_TAGS", None)
                save_code = os.getenv("WANDB_DISABLE_CODE", None)

                # environment variables get priority
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=run_name,
                    group=os.getenv("WANDB_RUN_GROUP"),
                    notes=os.getenv("WANDB_NOTES", None),
                    entity=os.getenv("WANDB_ENTITY", None),
                    id=os.getenv("WANDB_RUN_ID", None),
                    dir=os.getenv("WANDB_DIR", None),
                    tags=tags if tags is None else tags.split(","),
                    job_type=os.getenv("WANDB_JOB_TYPE", None),
                    mode=os.getenv("WANDB_MODE", None),
                    anonymous=os.getenv("WANDB_ANONYMOUS", None),
                    save_code=bool(save_code) if save_code is not None else save_code,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )

            # # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps),
                )

def compute_metrics(eval_pred, data_module, data_cfg):
    
    preds, labels = eval_pred  
    
    if data_cfg.problem_type == "single_label_classification":
        preds = np.array([data_module.id2label[i] for i in preds.argmax(-1)])
        labels = np.array([data_module.id2label[i] for i in labels])

    colwise_rmse = np.sqrt(np.mean((labels - preds) ** 2, axis=0))
    mean_rmse = np.mean(colwise_rmse)

    metrics = {}
    
    if data_cfg.problem_type == "multi_label_classification":
        for id_, label in data_module.id2label.items():
            metrics[f"{label}_rmse"] = colwise_rmse[id_]

        metrics["mcrmse"] = mean_rmse
    
    else:
        metrics[f"{data_cfg.label_col}_rmse"] = colwise_rmse
        

    return metrics


@dataclass
class PureMaskingDataCollator(DataCollatorForLanguageModeling):

    # Link to parent class
    # https://github.com/huggingface/transformers/blob/855dcae8bb743c3f8f0781742d7fa2fa3aaa3e22/src/transformers/data/data_collator.py#L607
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        
        batch["input_ids"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        label_name = "label" if "label" in examples[0].keys() else "labels"
        labels = [feature[label_name] for feature in examples] if label_name in examples[0].keys() else None

        if "label" in batch:
            del batch["label"]
        batch["labels"] = torch.tensor(labels)

        return batch

class MaskAugmentationTrainer(Trainer):
    """
    Not much changed. This is just so that the masking only happens during
    training and not during evaluation or predicting.
    """

    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        super().__init__(*args, **kwargs)


    def get_train_dataloader(self, *args, **kwargs):
        dataloader = super().get_train_dataloader(*args, **kwargs)

        dataloader.collate_fn = PureMaskingDataCollator(
            self.tokenizer, 
            mlm_probability=self.cfg.data.masking_prob, 
            pad_to_multiple_of=self.cfg.data.pad_multiple,
        )

        return dataloader
    
    
class SaveCallback(TrainerCallback):
    def __init__(self, threshold_score, metric_name, greater_is_better, weights_only=True) -> None:
        """
        After evaluation, if the `metric_name` value is higher than
        `min_score_to_save` the model will get saved.
        If `metric_name` value > `min_score_to_save`, then
        `metric_name` value becomes the new `min_score_to_save`.
        """
        super().__init__()

        self.threshold_score = threshold_score
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.weights_only = weights_only

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        metrics = kwargs.get("metrics")
        if metrics is None:
            raise ValueError("No metrics found for SaveCallback")

        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            raise KeyError(f"{self.metric_name} not found in metrics")
        
        surpassed_threshold =  metric_value > self.threshold_score if self.greater_is_better else metric_value < self.threshold_score
        if surpassed_threshold:
            logger.info(f"Saving model.")
            self.threshold_score = metric_value
            kwargs["model"].config.update({f"best_{self.metric_name}": metric_value})

            if self.weights_only:
                if "COCO" in str(kwargs["model"].config.__class__):
                    torch.save(kwargs["model"].state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
                else:
                    kwargs["model"].save_pretrained(args.output_dir)
                kwargs["model"].config.save_pretrained(args.output_dir)
                kwargs["tokenizer"].save_pretrained(args.output_dir)
            else:
                control.should_save = True
        else:
            logger.info("Not saving model.")
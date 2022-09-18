import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
import numpy as np
from transformers import (
    PreTrainedModel,
    AutoModel,
)
from transformers.utils import ModelOutput


class BaseModel(PreTrainedModel):
    
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.backbone = AutoModel.from_config(config)

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        
        if config.multisample_dropout:
            self.multisample_dropout = MultiSampleDropout(config.multisample_dropout)

        self.ln = nn.Identity()
        if config.output_layer_norm:
            self.ln = nn.LayerNorm(config.hidden_size)
            self._init_weights(self.ln)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self._init_weights(self.classifier)

    @staticmethod
    def get_shapes(problem_type, num_labels):
        """
        (shapes for logits, shapes for labels)
        """
        
        if problem_type == "single_label_classification":
            return (-1, num_labels), (-1, )
        if problem_type == "multi_label_classification":
            return (-1, num_labels), (-1, num_labels)
        
        return (-1,), (-1,)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        token_type_ids=None,
        **kwargs,
    ):

        token_type_ids = (
            {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
        )
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **token_type_ids,
            **kwargs,
        )[0]

        loss = None
        if labels is not None:
            
            loss_fct = LOSS_FUNCTIONS[self.config.loss_fn]

            
            if self.config.multisample_dropout:
                loss, logits = self.multisample_dropout(outputs[:, 0, :], self.classifier, labels, loss_fct, self.ln)
                
            else:

                logits = self.classifier(self.ln(self.dropout(outputs[:, 0, :])))
                logits_shape, labels_shape = self.get_shapes(self.config.problem_type, self.config.num_labels)
                loss = loss_fct(logits.view(*logits_shape), labels.view(*labels_shape))

        else:
            logits = self.classifier(self.ln(outputs))
            
            
        return ClassifierOutput(
            loss=loss, logits=logits,
        )

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None, **kwargs):
        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path)
        model = BaseModel(config)

        """
        If loading a local file, will first check if the state dict is to
        just the transformer model or to the whole model. 
        """
        if Path(model_name_or_path).is_dir():
            state_dict = torch.load(Path(model_name_or_path) / "pytorch_model.bin")
            
            if any(["BaseModel" in name for name in state_dict.keys()]):
                model.load_state_dict(state_dict)
            else:
                model.backbone.load_state_dict(state_dict)

            return model

        model.backbone = AutoModel.from_pretrained(model_name_or_path, config=config)
        
        return model
    
    
    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value


LOSS_FUNCTIONS = {
    "smoothl1":  nn.SmoothL1Loss(),
    "l1": nn.L1Loss(),
    "mse": nn.MSELoss(),
    "ce": nn.CrossEntropyLoss()
}


@dataclass
class ClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class MultiSampleDropout(nn.Module):
    def __init__(self, dropout_probs) -> None:
        super().__init__()

        self.dropouts = [nn.Dropout(p=p) for p in dropout_probs]

    def forward(self, hidden_states, linear, labels, loss_fn, layer_nm):
        # if not using output layer_nm, pass nn.Identity()

        logits = [linear(layer_nm(d(hidden_states))) for d in self.dropouts]

        losses = [loss_fn(log.view(-1, labels.size(1)), labels) for log in logits]

        logits = torch.mean(torch.stack(logits, dim=0), dim=0)
        loss = torch.mean(torch.stack(losses, dim=0), dim=0)

        return (loss, logits)
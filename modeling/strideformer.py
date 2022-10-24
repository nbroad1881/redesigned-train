""" PyTorch Strideformer Model """
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Iterator

import torch
from torch import nn
from transformers import PreTrainedModel, AutoModel, PretrainedConfig, AutoConfig
from transformers.modeling_outputs import ModelOutput


LOSS_FUNCTIONS = {
    "smoothl1":  nn.SmoothL1Loss(),
    "l1": nn.L1Loss(),
    "mse": nn.MSELoss(),
    "ce": nn.CrossEntropyLoss()
}

""" Strideformer model configuration """
from typing import Optional, List

from transformers import PretrainedConfig
from transformers.utils import logging



from itertools import chain
from typing import List, Dict

import torch
from transformers import PreTrainedTokenizerFast
class StrideformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Strideformer`]. It is used to instantiate a BStrideformerART
    model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        first_model_name_or_path (`str`, *optional*, defaults to `"sentence-transformers/all-MiniLM-L6-v2"`):
            The model name or path to the first model that is usually a pre-trained sentence transformer.
        freeze_first_model (`bool`, *optional*, defaults to `True`):
            If True, freeze the weights of the first model. Otherwise, train it as well.
        max_chunks (`int`, *optional*, defaults to 64)
            The maximum number of chunks the first model can take.
        hidden_size (`int`, *optional*, defaults to 384):
            The hidden size of the second model. It must match the first model's hidden size.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation when initializing weights using a normal distribution.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of layers for the second model.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the second model.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in second model.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the second model.
        layer_norm_eps (`float`, *optional*, defaults to 1e-7):
            The epsilon value in LayerNorm
        num_labels (`int`, *optional*, defaults to 2):
            The number of labels for the classifier.
    Example:
    ```python
    >>> from strideformer import StrideformerConfig, Strideformer
    >>> config = StrideformerConfig("sentence-transformers/all-MiniLM-L6-v2", max_chunks=128)
    >>> model = Strideformer(config)
    ```"""
    model_type: str = "strideformer"
    keys_to_ignore_at_inference: List = [
        "first_model_hidden_states",
        "second_model_hidden_states",
    ]

    def __init__(
        self,
        first_model_name_or_path: Optional[
            str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_first_model: Optional[bool] = True,
        max_chunks: Optional[int] = 64,
        hidden_size: Optional[int] = 384,
        initializer_range: Optional[float] = 0.02,
        num_hidden_layers: Optional[int] = 24,
        num_attention_heads: Optional[int] = 12,
        intermediate_size: Optional[int] = 4096,
        hidden_act: Optional[str] = "gelu",
        dropout: Optional[float] = 0.1,
        layer_norm_eps: Optional[float] = 1e-7,
        num_labels: Optional[int] = 2,
        **kwargs
    ):
        self.first_model_name_or_path = first_model_name_or_path
        self.freeze_first_model = freeze_first_model
        self.max_chunks = max_chunks
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_labels = num_labels

        super().__init__(
            num_labels=num_labels,
            **kwargs,
        )
@dataclass
class ClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class Strideformer(PreTrainedModel):

    config_class = StrideformerConfig
    supports_gradient_checkpointing = False

    def __init__(
        self,
        config: StrideformerConfig,
        first_model_config_path: Optional[Union[str, os.PathLike]] = None,
        first_init: Optional[bool] = False,
    ) -> None:
        """
        Initializes Strideformer model with random values.
        Use `from_pretrained` to load pretrained weights.
        Args:
            config (StrideformerConfig):
                Configuration file for this model. Holds the information for what
                pretrained model to use for the first model and all of the parameters
                for the hidden_size,
        """
        super().__init__(
            config,
        )
        self.config = config

        self.first_model_config = AutoConfig.from_pretrained(
            first_model_config_path or config.first_model_name_or_path
        )

        self.first_model = AutoModel.from_config(self.first_model_config)

        self.max_chunks = config.max_chunks

        if self.first_model_config.hidden_size != config.hidden_size:
            raise ValueError(
                "The hidden size of the first model \
                {self.first_model_config.hidden_size} needs to be the same \
                size as the hidden size of the second model {config.hidden_size}"
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
        )
        self.second_model = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self._init_weights(self.modules(), self.config.initializer_range)
        
        if first_init:
            self.first_model = AutoModel.from_pretrained(
                config.first_model_name_or_path
            )

    @staticmethod
    def mean_pooling(
        token_embeddings: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Mean pool across the `sequence_length` dimension. Assumes that token
        embeddings have shape `(batch_size, sequence_length, hidden_size)`.
        If batched, there can be pad tokens in the sequence.
        This will ignore padded outputs when doing mean pooling by using
        `attention_mask`.
        Args:
            token_embeddings (`torch.FloatTensor`):
                Embeddings to be averaged across the first dimension.
            attention_mask (`torch.LongTensor`):
                Attention mask for the embeddings. Used to ignore
                padd tokens from the averaging.
        Returns:
            `torch.FloatTensor`of shape `(batch_size, hidden_size)` that is
            `token_embeddings` averaged across the 1st dimension.
        """

        if attention_mask is not None:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        # this might be wrong
        return torch.sum(token_embeddings, 1) / torch.clamp(
            token_embeddings.sum(1), min=1e-9
        )
    
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = 1,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor]]:
        """
        Args:
            input_ids (`torch.Tensor`, *optional*, defaults to None):
                Indices of input sequence tokens in the vocabulary. These can be created
                using the corresponding tokenizer for the first model.
                Shape is `(num_chunks, sequence_length)` where `num_chunks` is `(batch_size*chunks_per_batch)`.
            token_type_ids (`torch.Tensor`, *optional*, defaults to None):
                Some models take token_type_ids. This comes from the tokenizer and gets
                passed to the first model.
            labels (`torch.FloatTensor` or `torch.Tensor`, *optional*, defaults to None):
                The true values. Used for loss calculation.
                Shape is `(batch_size, num_classes)` if multilabel,
                `(batch_size, 1)` for multiclass or regression.
            batch_size (`int`, *optional*, defaults to 1):
                If passing batched inputs, this specifies the shape of input for the second model.
                The first model will get input `(num_chunks, sequence_length)` wherewhere `num_chunks`
                is `(batch_size*chunks_per_batch)`. The output of the first model is `(num_chunks, hidden_size)`.
                This gets reshaped to `(batch_size, chunks_per_batch, hidden_size)`. This means that
                all document sequences must be tokenized to the same number of chunks.
        Returns:
            A `tuple` of `torch.Tensor` if `return_dict` is `False`.
            A `StrideformerOutput` object if `return_dict` is None or True.
            These containers hold values for loss, logits, and last hidden states for
            both models.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        token_type_ids = (
            {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
        )

        if self.config.freeze_first_model:
            # No gradients, no training, save memory
            with torch.no_grad():
                first_model_hidden_states = self.first_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **token_type_ids,
                )[0]
        else:
            first_model_hidden_states = self.first_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **token_type_ids,
            )[0]

        # mean pool last hidden state
        embeddings = self.mean_pooling(
            first_model_hidden_states, attention_mask=attention_mask
        )

        second_model_hidden_states = self.second_model(
            embeddings.reshape(batch_size, -1, self.config.hidden_size),
        )  # [batch_size, chunks_per_batch, hidden_size]

        # Classifier uses mean pooling to combine output embeddings into single embedding.
        second_model_chunk_logits = self.classifier(
            second_model_hidden_states
        )  # [batch_size, chunks_per_batch, num_labels]
        logits = second_model_chunk_logits.mean(dim=1)  # [batch_size, num_labels]

        loss = None
        if labels is not None:

            loss_fct = LOSS_FUNCTIONS[self.config.loss_fn]
            logits_shape, labels_shape = self.get_shapes(self.config.problem_type, self.config.num_labels)
            loss = loss_fct(logits.view(*logits_shape), labels.view(*labels_shape))
            

        if not return_dict:
            output = (logits, first_model_hidden_states, second_model_hidden_states)
            return ((loss,) + output) if loss is not None else output

        return ClassifierOutput(
            loss=loss,
            logits=logits,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs
    ):
        """
        Loads the model weights, the model.config, and the model.first_model_config from
        `pretrained_model_name_or_path`.
        Note: This is NOT the function that should be called the first time you create a
        Strideformer model. Even though the first model comes pretrained, the
        `Strideformer.from_pretrained` method is for loading after the second model
        has been trained.
        To load the model the first time, use `Strideformer(config, first_init=True)`
        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                A path to a *directory* containing model weights saved using
                [`Strideformer.save_pretrained`], e.g., `./my_model_directory/`.
        Returns:
            (Strideformer) with pretrained weights loaded.
        """
        first_model_path = str(
            Path(pretrained_model_name_or_path) / "first_model_config.json"
        )


        return super().from_pretrained(
            pretrained_model_name_or_path,
            first_model_config_path=first_model_path,
            first_init=False,
            *model_args,
            **kwargs,
        )

    def save_pretrained(
        self, save_directory: Optional[Union[str, os.PathLike]], *args, **kwargs
    ):
        """
        Saves the model, the model.config, and the model.first_model_config to
        `save_directory`.
        Args:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        self.first_model_config.save_pretrained(save_directory)
        first_model_config_path = Path(save_directory) / "config.json"
        first_model_config_path.replace(
            Path(save_directory) / "first_model_config.json"
        )

        super().save_pretrained(save_directory, *args, **kwargs)

    @staticmethod
    def _init_weights(modules: Iterator[nn.Module], std: float = 0.02) -> None:
        """
        Reinitializes every Linear, Embedding, and LayerNorm module provided.
        Args:
            modules (Iterator of `torch.nn.Module`)
                Iterator of modules to be initialized. Typically by calling Module.modules()
            std (`float`, *optional*, defaults to 0.02)
                Standard deviation for normally distributed weight initialization
        """
        for module in modules:
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


class StrideformerCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_chunks: int = 128, problem_type:str = "multi_label_classification"):
        """
        Loads a collator designed for Strideformer.
        Args:
            tokenizer (`PreTrainedTokenizerFast`):
                The tokenizer that corresponds with the first model in Strideformer.
            max_chunks (`int`, *optional*, defaults to 128):
                The maximum number of chunks that can be passed to the first model.
                This is to limit OOM errors.
        """
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks

    def __call__(self, features: List[Dict]):
        """
        Put features in a format that the model can use.
        Args:
            features (`List[Dict]`):
                The list will be as long as the batch size specified
                passed to the DataLoader.ffffffffffffffffffffffffffff
                Each element of features will have keys: input_ids, attention_mask, labels
                input_ids will be of shape [num_chunks, sequence_length]
                attention_mask will be of shape [num_chunks, sequence_length]
                label will be a single value if this is single_label_classification or regression
                It will be a list if multi_label_classification
        Returns:
            (dict): input_ids, attention_mask, labels to be passed to the model.
        """

        label_key = "label" if "label" in features[0] else "labels"

        ids = list(chain(*[x["input_ids"] for x in features]))
        mask = list(chain(*[x["attention_mask"] for x in features]))
        labels = [x[label_key] for x in features]

        longest_seq = max([len(x) for x in ids])

        ids = [x + [self.tokenizer.pad_token_id] * (longest_seq - len(x)) for x in ids]
        mask = [x + [0] * (longest_seq - len(x)) for x in mask]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long)[: self.max_chunks, :],
            "attention_mask": torch.tensor(mask, dtype=torch.long)[
                : self.max_chunks, :
            ],
            "labels": torch.tensor(labels, dtype=torch.float).squeeze(-1),
        }
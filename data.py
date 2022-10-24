from pathlib import Path
from itertools import chain
from functools import partial
from typing import Dict
from dataclasses import dataclass

import datasets
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import numpy as np


@dataclass
class DataModule:

    cfg: Dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError("Must pass config to DataModule")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.model_name_or_path,
        )

    def prepare_dataset(self) -> None:
        """
        Load in dataset and tokenize.

        If debugging, take small subset of the full dataset.
        """

        if self.cfg.data.load_from_disk:
            pass
        else:
            if self.cfg.data.data_files.train is not None:
                processor = LocalFileProcessor(
                    cfg=self.cfg,
                    tokenizer=self.tokenizer,
                )
            else:
                raise ValueError("Must specify local train file")

            self.raw_dataset, self.tokenized_dataset = processor.prepare_dataset()

            self.label2id = processor.get_label2id()
            self.id2label = {i: l for l, i in self.label2id.items()}

            self.fold_idxs = []

            temp_df = self.raw_dataset["train"].to_pandas().reset_index(drop=True)
            for k in range(self.cfg.data.kfolds):
                self.fold_idxs.append(temp_df[temp_df["kfold"]==k].index.tolist())


    def get_train_dataset(self, fold: int) -> datasets.Dataset:
        train_idxs = list(chain(*[idxs for f, idxs in enumerate(self.fold_idxs) if f != fold]))
        return self.tokenized_dataset["train"].select(train_idxs)

    def get_eval_dataset(self, fold: int) -> datasets.Dataset:
        return self.tokenized_dataset["train"].select(self.fold_idxs[fold])
        


class LocalFileProcessor:
    """
    Can load csv, json, or parquet files that are on local storage.
    """

    def __init__(self, cfg, tokenizer):
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

    def prepare_dataset(self):

        # get ending of file (csv, json, parquet)
        filetype = Path(self.cfg.data.data_files["train"][0]).suffix
        filetype = filetype.lstrip(".")  # suffix keeps period

        if filetype not in {"csv", "json", "parquet"}:
            raise ValueError(
                f"Files should end in 'csv', 'json', or 'parquet', not {filetype}."
            )

        data_files = OmegaConf.to_container(self.cfg.data.data_files)
        raw_dataset = load_dataset(filetype, data_files=data_files)

        # Limit the number of rows, if desired
        if self.cfg.data.n_rows is not None and self.cfg.data.n_rows > 0:
            for split in raw_dataset:
                max_split_samples = min(self.cfg.data.n_rows, len(raw_dataset[split]))
                raw_dataset[split] = raw_dataset[split].select(range(max_split_samples))

        cols = raw_dataset["train"].column_names
        self.set_label2id(raw_dataset["train"])

        tokenized_dataset = raw_dataset.map(
            partial(
                tokenize,
                tokenizer=self.tokenizer,
                max_length=self.cfg.data.max_seq_length,
                stride=self.cfg.data.stride,
                text_col=self.cfg.data.text_col,
                text_pair_col=self.cfg.data.text_pair_col,
                label_col=self.cfg.data.label_col,
                label2id=self.label2id,
            ),
            batched=self.cfg.data.stride in {None, 0},
            batch_size=self.cfg.data.map_batch_size,
            num_proc=self.cfg.num_proc,
            remove_columns=cols,
        )

        return raw_dataset, tokenized_dataset

    def set_label2id(self, train_dataset: Dataset):
        
        if isinstance(self.cfg.data.label_col, str):
            label_col = self.cfg.data.label_col
            labels = train_dataset.unique(self.cfg.data.label_col)
            labels = sorted(labels)
        else:
            label_cols = OmegaConf.to_container(self.cfg.data.label_col)
            labels = sorted(label_cols)            

        self.label2id = {label: i for i, label in enumerate(labels)}

    def get_label2id(self):
        """
        Must be called after `set_label2id`
        """
        return self.label2id


def tokenize(
    examples,
    tokenizer,
    max_length,
    stride=None,
    text_col="text",
    text_pair_col=None,
    label_col="label",
    label2id=None,
):
    tokenizer_kwargs = {
        "padding": False,
    }

    # If stride is not None, using sbert approach
    if stride is not None and stride > 0:
        tokenizer_kwargs.update(
            {
                "padding": True,
                "stride": stride,
                "return_overflowing_tokens": True,
            }
        )

    texts = [examples[text_col]]
    if text_pair_col is not None:
        texts.append(examples[text_pair_col])

    tokenized = tokenizer(
        *texts,
        truncation=True,
        max_length=max_length,
        **tokenizer_kwargs,
    )

    # multi-label
    if not isinstance(label_col, str):
        
        num_rows = 1 if stride is not None else len(examples[text_col])
        labels = np.zeros(shape=(num_rows, len(label_col)), dtype=np.float32)

        label_names = sorted(label_col)

        for col, l in enumerate(label_names):
            labels[:, col] = examples[l]

    else:
        labels = [label2id[l] for l in examples[label_col]]

    tokenized["labels"] = labels
    

    return tokenized
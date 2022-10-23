import spacy
from datasets import Dataset

from transformers import DebertaV2Model

NO_LABEL = "<none>"

class SpacyForEmbeddings:

    def __init__(self, model_name="en_core_web_sm", ner=True, pos=True, dep=True, *args, **kwargs):
        self.nlp = spacy.load(model_name)
        self.types = []
        if ner: self.types.append("ner")
        if pos: self.types.append("pos")
        if dep: self.types.append("dep")
        ALL_LABELS = {
            "ner": [],
            "pos": [],
            "dep": []
        }

        

        self.labels = {name: ALL_LABELS[name] for name in self.types}

        for key in self.labels.keys():
            self.labels[key].append(NO_LABEL)

        self.labels = {name: sorted(labels) for name, labels in self.labels.items()}

        self.label2id = {
            name: {l:i for i, l in enumerate(self.labels[name])} for name in self.labels.keys()
        }
        

    def get_labels_and_offsets(self, texts, num_proc=1):
        """
        Run spacy model on each text.
        Get pos, ner, dep labels, including offsets
        """

        docs = self.nlp.pipe(texts)

        ds = Dataset.from_dict({"doc": docs})

        def _get_labels_and_offsets(doc):
            example_labels = {name: [] for name in self.labels.keys()}
            offsets = []

            for token in doc:
                for name in self.labels.keys():
                    label = getattr(token, f"{name}_")
                    example_labels[name].append(self.label2id[name][label])
                offsets.append((token.start, token.end))

            return {
                **example_labels,
                "spacy_offsets": offsets,
            }
        

        return ds.map(_get_labels_and_offsets, batched=False, num_proc=num_proc, remove_columns=["doc"])

    @staticmethod
    def tokenize(example, tokenizer, label2id, max_length):
        """
        Tokenize the text and return input_ids, attention_mask, ner_ids, pos_ids, and dep_ids.
        """
        tokenized = tokenizer(
            example["text"],
            return_offset_mappings=True,
            truncation=True,
            max_length=max_length,
        )

        ids = {name:[] for name in label2id.keys()}
        offsets_idx = 0
        for s, e in tokenized["offsets_mapping"]:

            if s is None:
                for name in ids.keys():
                    ids[name].append(label2id[name][NO_LABEL])
            else:
                if offsets_idx < len(example["spacy_offsets"]):
                    spacy_start, spacy_end = example["spacy_offsets"][offsets_idx]
                    while spacy_end < s and offsets_idx < len(example["spacy_offsets"]):
                        offsets_idx += 1
                        spacy_start, spacy_end = example["spacy_offsets"][offsets_idx]

                    for name in ids.keys():
                        label = example[name][offsets_idx]
                        ids[name].append(label2id[name][label])
                else:
                    for name in ids.keys():
                        ids[name].append(label2id[name][NO_LABEL])

        for name, ids in ids.items():
            tokenized[f"{name}_ids"] = ids

            assert len(ids) == len(tokenized["input_ids"])

        return tokenized



def forward(self, input_ids, attention_mask, labels, ner_ids=None, pos_ids=None, dep_ids=None):

    ner_embeds = self.ner_embeddings[ner_ids]
    pos_embeds = self.pos_embeddings[pos_ids]
    pos_embeds = self.pos_embeddings[pos_ids]
    
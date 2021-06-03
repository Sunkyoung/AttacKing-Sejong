import argparse
import json
import os
from typing import List, Optional

import torch
from Overrides import overrides
from torch.utils.data import TensorDataset


class InputExample(object):
    """A single example"""

    def __init__(
        self, guid: str, label: str, first_seq: str, second_seq: Optional[str] = None
    ):
        self.guid = guid
        self.label = label
        self.first_seq = first_seq
        self.second_seq = second_seq


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids: List[int],
        input_mask: List[int],
        label_id: Optional[int] = None,
        segment_ids: Optional[List[int]] = None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class OutputFeatures(object):
    def __init__(self, label_id: int, first_seq: str, second_seq: Optional[str] = None):
        self.label_id = label_id
        self.first_seq = first_seq
        self.second_seq = second_seq
        self.final_text = None
        self.similarity = 0.0
        self.query_length = 0
        self.num_changes = 0
        self.changes = []
        self.success_indication = None


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_data(self, data_dir: str) -> TensorDataset:
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_data(self, data_dir: str) -> TensorDataset:
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self) -> List[str]:
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _convert_examples_to_features(
        self,
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer,
        do_whitespace_tokenize,
    ) -> TensorDataset:

        """Loads a data file into a list of `InputBatch`s."""
        features = []
        label_map = {label: i for i, label in enumerate(label_list)}

        for example in examples:
            if do_whitespace_tokenize:
                first_tokens = example.first_seq.strip().split()
                second_tokens = example.second_seq.strip().split()
            else:
                first_tokens = tokenizer.tokenize(example.first_seq)
                second_tokens = tokenizer.tokenize(example.second_seq)

            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(first_tokens, second_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + first_tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            tokens += second_tokens + ["[SEP]"]
            segment_ids += [1] * (len(second_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    label_id=label_id,
                    segment_ids=segment_ids,
                )
            )

        return self._convert_to_tensordata(features)

    def _truncate_seq_pair(
        self, first_tokens: List[str], second_tokens: List[str], max_length: int
    ):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(first_tokens) + len(second_tokens)
            if total_length <= max_length:
                break
            if len(first_tokens) > len(second_tokens):
                first_tokens.pop()
            else:
                second_tokens.pop()

    def _convert_to_tensordata(self, features: List[InputFeatures]) -> TensorDataset:
        """ Convert to Tensors and build dataset """
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )

    @classmethod
    def _read_txt(cls, input_file: str):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            lines = f.readlines()
            data = []
            for line in lines[1:]:  # remove header
                data.extend(line.split("\t"))
            return data

    @classmethod
    def _read_json(cls, input_file: str):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            raw_data = json.load(f)
            data = []
            for rd in raw_data:
                data.append([rd["guid"], rd["title"], rd["label"]])
            return data


class NsmcProcessor(DataProcessor):
    def get_dev_data(self, data_dir: str) -> TensorDataset:
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "rating_test.txt"))
        )

    def get_labels(self) -> List[str]:
        return ["0", "1"]

    def _create_examples(self, data):
        examples = []
        for d in data:
            guid, first_seq, label = d
            examples.append(InputExample(guid, first_seq, label))
        return examples


class YnatProcessor(DataProcessor):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = 512

    def get_train_data(self, data_dir: str) -> List[InputExample]:
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "ynat-v1_train.json"))
        )

    def get_dev_data(self, data_dir: str) -> List[InputExample]:
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "ynat-v1_dev.json"))
        )

    def get_labels(self) -> List[str]:
        return ["정치", "사회", "경제", "세계", "생활문화", "IT과학", "스포츠"]

    def get_features(self, examples) -> List[InputFeatures]:
        return self.convert_examples_to_features(
            examples,
            self.get_labels(),
            self.args.max_seq_length,
            self.tokenizer,
            self.args.whitespace_tokenize,
        )

    def get_tensor(self, feature) -> TensorDataset:
        return self._convert_to_tensordata(feature)

    def _create_examples(self, data):
        examples = []
        for d in data:
            guid, first_seq, label = d
            examples.append(InputExample(guid, first_seq, label))
        return examples

    @classmethod
    def _get_masked(self, feature: TensorDataset) -> TensorDataset:
        masked_inputs = []
        for i in range(1, len(feature.input_ids)-1):
            masked_ids = feature.input_ids
            masked_ids[i] = self.tokenizer.mask_token_id
            masked_inputs.append(masked_ids)
        return self._convert_to_tensordata(masked_inputs)

    @overrides
    def _convert_examples_to_features(
        self,
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer,
        do_whitespace_tokenize,
    ) -> TensorDataset:

        """Loads a data file into a list of `InputBatch`s."""
        features = []
        label_map = {label: i for i, label in enumerate(label_list)}

        for example in examples:
            if do_whitespace_tokenize:
                first_tokens = example.first_seq.strip().split()
            else:  # white space 기준 만으로 token화
                first_tokens = tokenizer.tokenize(example.first_seq)

            # Account for [CLS] and [SEP] with "- 2"
            if len(first_tokens) > max_seq_length - 2:
                first_tokens = first_tokens[: (max_seq_length - 2)]

            tokens = ["[CLS]"] + first_tokens + ["[SEP]"]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            label_id = [0.0] * len(label_list)
            label_id[label_map[example.label]] += 1

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    label_id=label_id,
                    segment_ids=None,
                )
            )

        return features

    @overrides
    def _convert_to_tensordata(self, feature) -> torch.tensor:
        """ Convert to Tensors and build dataset """
        return torch.tensor([ f for f in feature ], dtype=torch.long)

    @staticmethod
    def add_specific_args(
        parser: argparse.ArgumentParser, root_dir: str
    ) -> argparse.ArgumentParser:
        parser.add_argument(
            "--max-seq-length",
            default=128,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded.",
        )
        parser.add_argument("--do-whitespace-tokenize", action="store_true")

        return parser

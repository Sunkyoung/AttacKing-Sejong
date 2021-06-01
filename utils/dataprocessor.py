import argparse
import json
import os
from typing import List, Optional

import torch
from torch.utils.data import TensorDataset


class InputExample(object):
    """A single example"""

    def __init__(self, guid: str, label: str, first_seq: str, second_seq: Optional[str] = None):
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
        segment_ids:  Optional[List[int]]=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_target_data(self, data_dir: str) -> TensorDataset:
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self) -> List[str]:
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

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
                data.append([rd["guid"], rd["label"], rd["title"]])
            return data


class NsmcProcessor(DataProcessor):
    def get_target_data(self, data_dir: str) -> TensorDataset:
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "rating.txt"))
        )

    def get_labels(self) -> List[str]:
        return ["0", "1"]

    def _create_examples(self, data):
        examples = []
        for d in data:
            examples.append(InputExample(guid=d[0], label=d[2], first_seq=d[1]))
        return examples


class YnatProcessor(DataProcessor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def get_target_data(self, data_dir: str) -> List[InputExample]:
        return self._create_dataset(
            self._read_json(os.path.join(data_dir, "ynat-v1_dev.json"))
        )

    def get_labels(self) -> List[str]:
        return ["정치", "사회", "경제", "세계", "생활문화", "IT과학", "스포츠"]

    def _create_dataset(self, data):
        return self._create_examples(data)

    def _create_examples(self, data):
        examples = []
        for d in data:
            examples.append(InputExample(guid=d[0], label=d[1], first_seq=d[2]))
        return self._convert_examples_to_features(
            examples, self.get_labels, self.args.max_seq_length, self.tokenizer
        )

    def _convert_examples_to_features(
        self,
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer=None,
    ) -> TensorDataset:

        """Loads a data file into a list of `InputBatch`s."""
        features = []
        label_map = {label: i for i, label in enumerate(label_list)}

        for example in examples:
            if tokenizer:
                first_tokens = tokenizer.tokenize(example.first_seq)
            else:  # white space 기준 만으로 token화
                first_tokens = example.first_seq.strip().split()

            second_tokens = None
            if example.second_seq:
                second_tokens = tokenizer.tokenize(example.second_seq)
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(first_tokens, second_tokens, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(first_tokens) > max_seq_length - 2:
                    first_tokens = first_tokens[: (max_seq_length - 2)]

            tokens = ["[CLS]"] + first_tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if second_tokens:
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
                    segment_ids=segment_ids,
                    label_id=label_id,
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

    @staticmethod
    def add_specific_args(
        parser: argparse.ArgumentParser, root_dir: str
    ) -> argparse.ArgumentParser:
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded.",
        )

        return parser

import os

class InputExample(object):
    """A single training/test example"""

    def __init__(self, guid, label, first_sequence, second_sequence=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            label: string. The label of the example.
            first_sequence : string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            second_sequence : (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.label = label
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_target_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_txt(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            lines = f.readlines()
            data = []
            for line in lines[1:] : # remove header
                data.extend(line.split('\t'))
            return data

class NsmcProcessor(DataProcessor):
    def get_target_examples(self, data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "rating.txt")))

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, data):
        examples = []
        for d in data:
            examples.append(
                InputExample(guid = d[0], label = d[2], first_sequence = d[1]))
        return examples
    
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    features = []

    for example in examples:
        first_tokens = tokenizer.tokenize     
        second_tokens = None
        if example.second_sequence:
            second_tokens = tokenizer.tokenize(second_tokens)        

            # Modifies `first_tokens` and `second_tokens` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(first_tokens, second_tokens, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(first_tokens) > max_seq_length - 2:
                first_tokens = first_tokens[:(max_seq_length-2)]

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
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(first_tokens, second_tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(first_tokens) + len(second_tokens)
        if total_length <= max_length:
            break
        if len(first_tokens) > len(second_tokens):
            first_tokens.pop()
        else:
            second_tokens.pop()
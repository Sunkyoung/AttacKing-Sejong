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
    
    def __init__(self, input_ids, input_mask, segment_ids, label_id, sentence_id=None):
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

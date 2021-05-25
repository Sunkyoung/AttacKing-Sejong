import os

class InputExample(object):
    """A single training/test example"""

    def __init__(self, guid, label, seq1, seq2=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            label: string. The label of the example.
            seq1 : string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            seq2 : (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.label = label
        self.seq1 = seq1
        self.seq2 = seq2


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
            guid = d[0]
            seq1 = d[1]
            label = d[2]
            examples.append(
                InputExample(guid=guid, label=label, seq1=seq1))
        return examples
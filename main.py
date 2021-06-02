import argparse
import os

from utils.dataprocessor import YnatProcessor
from utils.attack import run_attack
from transformers import (AutoConfig, AutoModel, AutoModelForMaskedLM,
                          AutoTokenizer)

def add_general_args(
    parser: argparse.ArgumentParser, root_dir: str
) -> argparse.ArgumentParser:
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data path. Should contain the .json files (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    return parser


def main(args):
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser, os.getcwd())
    parser = YnatProcessor.add_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = YnatProcessor(args, tokenizer)
    target_examples = processor.get_dev_data(args.data_dir)
    target_features = processor.get_features(target_examples)
    num_labels = len(processor.get_labels())

    mlm_config = AutoConfig.from_pretrained(model_name)
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name, config=mlm_config)
    mlm_model.to("cuda")

    finetuned_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    finetuned_model = AutoModel.from_pretrained(model_name)
    finetuned_model = AutoModelForMaskedLM.from_pretrained(
        model_name, config=finetuned_config
    )
    finetuned_model.to("cuda")

    run_attack(processor, target_examples, target_features, mlm_model, finetuned_model)

if __name__ == "__main__":
    main()
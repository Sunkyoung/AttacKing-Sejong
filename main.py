import argparse
import os
import json

from utils.dataprocessor import YnatProcessor
from utils.attack import run_attack
from transformers import (AutoConfig, AutoModel, AutoModelForMaskedLM,
                          AutoTokenizer)

def dump_features(features, output_dir):
    output_file = 'attacked_result.json'
    outputs = []
    for feature in features:
        outputs.append({
                        'success_indication': feature.success_indication,
                        'target_sequence': feature.first_seq,
                        'num_tokens': len(feature.first_seq.split(' ')),
                        'label': feature.label_id,
                        'query_length': feature.query_length,
                        'num_changes': feature.num_changes,
                        'changes': feature.changes,
                        'attacked_text': feature.final_text,
                        })
    json.dump(outputs, open(os.join.dir(output_dir, output_file), 'w'), indent=4, ensure_ascii=False)

    print('Finished dump')

def add_general_args(
    parser: argparse.ArgumentParser, root_dir: str
) -> argparse.ArgumentParser:
    # Required parameters
    parser.add_argument(
        "--data-dir",
        default=None,
        type=str,
        required=True,
        help="The input data path. Should contain the .json files (or other data files) for the task.",
    )
    parser.add_argument(
        "--output-dir",
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

    output_features = []
    for example, feature in zip(target_examples, target_features):
        output = run_attack(processor, example, feature, mlm_model, finetuned_model)
        output_features.append(output)
    
    dump_features(output_features, args.output_dir)


if __name__ == "__main__":
    main()
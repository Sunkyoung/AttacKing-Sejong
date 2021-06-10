import argparse
import json
import os
import time
import torch

from transformers import (AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification,
                          AutoTokenizer)
from utils.dataprocessor import DataProcessor, YNATProcessor
from utils import attack
from utils.attack import run_attack
from utils.attack_original import run_attack as original_run_attack 

def dump_features(features, output_dir):
    output_file = "attacked_result_counterfit_epoch2.json"
    outputs = []
    for feature in features:
        outputs.append(
            {
                "success_indication": feature.success_indication,
                "target_sequence": feature.first_seq,
                "num_tokens": len(feature.first_seq.split(" ")),
                "original_label": feature.label_id,
                "change_label": feature.final_label if feature.final_label else feature.label_id ,
                "query_length": feature.query_length,
                "num_changes": feature.num_changes,
                "changes": feature.changes,
                "attacked_text": feature.final_text,
            }
        )
    json.dump(
        outputs,
        open(os.path.join(output_dir, output_file), "w"),
        indent=4,
        ensure_ascii=False,
    )

    print("Finished dump")

def calculate_accuracy(features):
    total = len(features)
    cnt_pred_fail = 0
    cnt_success = 0
    for feature in features:
        if feature.success_indication == "Predict fail":
            cnt_pred_fail += 1
        if feature.success_indication == "Attack success":
            cnt_success += 1
    orig_acc = (1 - cnt_pred_fail / total) * 100
    attack_acc = orig_acc - (cnt_success / total * 100)
    print(f'original accuarcy : {orig_acc}')
    print(f'attacked accuarcy : {attack_acc}')


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
    parser.add_argument(
        "--run-BertAttack-original",
        action='store_true',
        help="If True then Attack with white space wise otherwise Attack with subword wise",
    )
    parser.add_argument(
        "--use-bpe",
        action='store_true',
        help="If True then use a bpe word for getting pair of subwords substitute"
    )
    parser.add_argument(
        "--finetuned-model-path",
        type=str,
        required=True,
        help="Path of finetnued model",
    )

    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser, os.getcwd())
    parser = DataProcessor.add_specific_args(parser, os.getcwd())
    parser = attack.add_specific_args(parser, os.getcwd())
    args = parser.parse_args()        

    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = YNATProcessor(args, tokenizer)
    target_examples = processor.get_dev_data(args.data_dir)
    target_features = processor.get_features(target_examples)
    num_labels = len(processor.get_labels())

    pretrained_config = AutoConfig.from_pretrained(model_name)
    pretrained_model = AutoModelForMaskedLM.from_pretrained(
        model_name, config=pretrained_config
    )
    pretrained_model.to("cuda")

    finetuned_model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)
    finetuned_model.to("cuda")
    model_state, _ = torch.load(args.finetuned_model_path)
    finetuned_model.load_state_dict(model_state, strict=False)

    start = time.time()
    output_features = []
    for example, feature in zip(target_examples, target_features):
        if args.run_BertAttack_original:
            output = original_run_attack(
                args, processor, example, feature, pretrained_model, finetuned_model
            )
        output = run_attack(
            args, processor, example, feature, pretrained_model, finetuned_model
        )
        output_features.append(output)
    print('Total execution time : ', time.time()-start)
    calculate_accuracy(output_features)
    dump_features(output_features, args.output_dir)


if __name__ == "__main__":
    main()

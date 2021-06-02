from enum import Enum
from typing import List
from torch.utils.data import (DataLoader, SequentialSampler)
from utils.dataprocessor import YnatProcessor
import torch

# feature.success = 1 : exceed ratio of word change
# feature.success = 2 : Fail to attack
# feature.success = 3 : Origin prediction Fail
# feature.success = 4 : attack success with substitutes

class SuccessIndicator(Enum):
    large_change = 1
    predict_fail = 2
    attack_fail = 3
    attack_success = 4

def get_important_scores(processor, target_features, tgt_model, orig_prob, orig_label, orig_probs, batch_size):
    masked_features = processor._get_masked(target_features)
    eval_sampler = SequentialSampler(masked_features)
    eval_dataloader = DataLoader(masked_features, sampler=eval_sampler, batch_size=batch_size)

    leave_1_probs = []
    with torch.no_grad():
        for batch in eval_dataloader:
            masked_input, _, _ = batch
            # bs = masked_input.size(0)
            leave_1_prob_batch = tgt_model(masked_input)[0]  # B num-label
            leave_1_probs.append(leave_1_prob_batch)
    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (
        (
            orig_prob
            - leave_1_probs[
                :, orig_label
            ]  # Difference between original logit output and 1 masked logit output
            + (  # Add score which In case the results change.
                leave_1_probs_argmax != orig_label
            ).float()
            * (
                leave_1_probs.max(dim=-1)[0]
                - torch.index_select(orig_probs, 0, leave_1_probs_argmax)
            )
        )
        .data.cpu()
        .numpy()
    )

    return import_scores


def attack(
    feature,
    tgt_model,
    mlm_model,
    tokenizer,
    k,
    batch_size,
    max_length=512,
    cos_mat=None,
    w2i={},
    i2w={},
    use_bpe=1,
    threshold_pred_score=0.3,
):
    # MLM-process

    # original label
    inputs = tokenizer.encode_plus(
        feature.seq,
        None,
        add_special_tokens=True,
        max_length=max_length,
    )
    input_ids, token_type_ids = torch.tensor(inputs["input_ids"]), torch.tensor(
        inputs["token_type_ids"]
    )
    attention_mask = torch.tensor([1] * len(input_ids))
    seq_len = input_ids.size(0)
    orig_probs = tgt_model(
        input_ids.unsqueeze(0).to("cuda"),
        attention_mask.unsqueeze(0).to("cuda"),
        token_type_ids.unsqueeze(0).to("cuda"),
    )[0].squeeze()
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs)
    current_prob = orig_probs.max()

    if orig_label != feature.label:
        feature.success = 3
        return feature

    sub_words = ["[CLS]"] + sub_words[: max_length - 2] + ["[SEP]"]
    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
    word_predictions = mlm_model(input_ids_.to("cuda"))[
        0
    ].squeeze()  # seq-len(sub) vocab.
    word_pred_scores_all, word_predictions = torch.topk(
        word_predictions, k, -1
    )  # seq-len k  #top k prediction

    word_predictions = word_predictions[1 : len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1 : len(sub_words) + 1, :]

    important_scores = get_important_scores(
        words,
        tgt_model,
        current_prob,
        orig_label,
        orig_probs,  # get important score
        tokenizer,
        batch_size,
        max_length,
    )
    feature.query += int(len(words))
    # sort by important score
    list_of_index = sorted(
        enumerate(important_scores), key=lambda x: x[1], reverse=True
    )  # sort the important score and index
    # print(list_of_index)
    # => [(59, 0.00014871359), (58, 0.00011396408), (60, 0.00010085106), .... ]      [(index, Importacne score), ....]
    final_words = copy.deepcopy(words)  # whole word corpus from mlm

    for top_index in list_of_index:
        # limit ratio of word change
        if feature.change > int(0.4 * (len(words))):
            feature.success = 1  # exceed
            return feature

        tgt_word = words[top_index[0]]
        if tgt_word in filter_words:
            continue
        if keys[top_index[0]][0] > max_length - 2:
            continue

        substitutes = word_predictions[
            keys[top_index[0]][0] : keys[top_index[0]][1]
        ]  # L, k
        word_pred_scores = word_pred_scores_all[
            keys[top_index[0]][0] : keys[top_index[0]][1]
        ]

        substitutes = get_substitues(
            substitutes,
            tokenizer,
            mlm_model,
            use_bpe,
            word_pred_scores,
            threshold_pred_score,
        )

        most_gap = 0.0
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_

            if substitute == tgt_word:
                continue  # filter out original word
            if "##" in substitute:
                continue  # filter out sub-word

            if substitute in filter_words:
                continue
            if substitute in w2i and tgt_word in w2i:
                if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                    continue

            # Replace word and check whether the attack is successful
            temp_replace = final_words
            temp_replace[top_index[0]] = substitute  # replace token
            temp_text = tokenizer.convert_tokens_to_string(temp_replace)
            inputs = tokenizer.encode_plus(
                temp_text,
                None,
                add_special_tokens=True,
                max_length=max_length,
            )
            input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to("cuda")
            seq_len = input_ids.size(1)
            temp_prob = tgt_model(input_ids)[0].squeeze()
            feature.query += 1
            temp_prob = torch.softmax(temp_prob, -1)
            temp_label = torch.argmax(temp_prob)
            # Success
            if temp_label != orig_label:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = 4

                return feature
            else:

                label_prob = temp_prob[orig_label]
                gap = current_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate

    feature.final_adverse = tokenizer.convert_tokens_to_string(final_words)
    feature.success = 2
    return feature

def run_attack(processor, target_features, mlm_model, finetuned_model):
    get_important_scores(processor, target_features)



with torch.no_grad():
    for index, feature in enumerate(features[start:end]):
        seq_a, label = feature
        feat = Feature(seq_a, label)
        print("\r number {:d} ".format(index) + tgt_path, end="")
        # print(feat.seq[:100], feat.label)
        feat = attack(
            feat,
            tgt_model,
            mlm_model,
            tokenizer,
            k,
            batch_size=32,
            max_length=512,
            cos_mat=cos_mat,
            w2i=w2i,
            i2w=i2w,
            use_bpe=use_bpe,
            threshold_pred_score=threshold_pred_score,
        )

        print(feat.changes, feat.change, feat.query, feat.success)
        if feat.success > 2:
            print("success", end="")
        else:
            print("failed", end="")
        features_output.append(feat)

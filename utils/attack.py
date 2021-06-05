import argparse

import torch
from torch.utils.data import DataLoader, SequentialSampler

from utils.dataprocessor import OutputFeatures


def get_important_scores(
    processor,
    target_features,
    tgt_model,
    current_prob,
    orig_label,
    pred_logit,
    batch_size,
):
    masked_features = processor._get_masked(target_features)
    eval_sampler = SequentialSampler(masked_features)
    eval_dataloader = DataLoader(
        masked_features, sampler=eval_sampler, batch_size=batch_size
    )

    leave_1_probs = []
    with torch.no_grad():
        for batch in eval_dataloader:
            masked_input = batch.to('cuda')
            # bs = masked_input.size(0)
            leave_1_prob_batch = tgt_model(masked_input)[0]  # B num-label
            leave_1_probs.append(leave_1_prob_batch)
    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (
        (
            current_prob
            - leave_1_probs[
                :, orig_label
            ]  # Difference between original logit output and 1 masked logit output
            + (  # Add score which In case the results change.
                leave_1_probs_argmax != orig_label
            ).float()
            * (
                leave_1_probs.max(dim=-1)[0]
                - torch.index_select(pred_logit, 0, leave_1_probs_argmax)
            )
        )
        .data.cpu()
        .numpy()
    )

    return import_scores


def run_attack(args, processor, example, feature, pretrained_model, finetuned_model):
    output = OutputFeatures(label_id=example.label, first_seq=example.first_seq)
    input_tensor = processor.get_tensor(feature.input_ids).unsqueeze(0).to('cuda')
    input_mask_tensor = processor.get_tensor(feature.input_mask).unsqueeze(0).to('cuda')
    # print(input_tensor.shape)
    with torch.no_grad():
        logit = finetuned_model(
            input_tensor, token_type_ids=None, attention_mask=input_mask_tensor
        )
        word_predictions = pretrained_model(input_tensor)[0].detach()

    pred_logit = logit[0]
    pred_logit = pred_logit.detach().cpu()  # orig prob -> pred logit 으로 변경
    pred_label = torch.argmax(
        pred_logit, dim=1
    ).flatten()  # orig label -> pred label 으로 변경
    current_prob = pred_logit.max()

    if pred_label != feature.label_id:
        output.success_indication = "Predict fail"
        return output

    # word prediction은 MLM 모델에서 각 토큰 당 예측 값을 뽑음
    word_predictions = word_predictions[1:-1, :]  # except  [CLS], [SEP]
    # Top-K 개를 뽑아서 가장 높은 스코어 순으로 정렬하며, 가장 plausible 한 예측값들의 모음
    # torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
    word_pred_score, word_pred_idx = torch.topk(
        word_predictions, args.top_k, -1
    )  # seq-len k  #top k prediction

    important_score = get_important_scores(
        processor,
        feature,
        finetuned_model,
        current_prob,
        pred_label,
        pred_logit,
        args.batch_size,
    )

    # important_score 다음 프로세스 (TBD)
    # legacy code
    # output.query_length += int(len(words))
    # # sort by important score
    # list_of_index = sorted(
    #     enumerate(important_scores), key=lambda x: x[1], reverse=True
    # )  # sort the important score and index
    # # print(list_of_index)
    # # => [(59, 0.00014871359), (58, 0.00011396408), (60, 0.00010085106), .... ]      [(index, Importacne score), ....]
    # final_words = copy.deepcopy(words)  # whole word corpus from mlm

    # for top_index in list_of_index:
    #     # limit ratio of word change
    #     if output.num_changes > int(args.change_ratio_limit * (len(words))):
    #         output.success_indication = 'exceed change ratio limit'  # exceed
    #         return output

    #     tgt_word = words[top_index[0]]
    #     if tgt_word in filter_words:
    #         continue
    #     if keys[top_index[0]][0] > max_length - 2:
    #         continue

    #     substitutes = word_predictions[
    #         keys[top_index[0]][0] : keys[top_index[0]][1]
    #     ]  # L, k
    #     word_pred_scores = word_pred_scores_all[
    #         keys[top_index[0]][0] : keys[top_index[0]][1]
    #     ]

    #     substitutes = get_substitues(
    #         substitutes,
    #         tokenizer,
    #         mlm_model,
    #         use_bpe,
    #         word_pred_scores,
    #         threshold_pred_score,
    #     )

    #     most_gap = 0.0
    #     candidate = None

    #     for substitute_ in substitutes:
    #         substitute = substitute_

    #         if substitute == tgt_word:
    #             continue  # filter out original word
    #         if "##" in substitute:
    #             continue  # filter out sub-word

    #         if substitute in filter_words:
    #             continue
    #         if substitute in w2i and tgt_word in w2i:
    #             if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
    #                 continue

    #         # Replace word and check whether the attack is successful
    #         temp_replace = final_words
    #         temp_replace[top_index[0]] = substitute  # replace token
    #         temp_text = tokenizer.convert_tokens_to_string(temp_replace)
    #         inputs = tokenizer.encode_plus(
    #             temp_text,
    #             None,
    #             add_special_tokens=True,
    #             max_length=max_length,
    #         )
    #         input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to("cuda")
    #         seq_len = input_ids.size(1)
    #         temp_prob = tgt_model(input_ids)[0].squeeze()
    #         output.query_length += 1
    #         temp_prob = torch.softmax(temp_prob, -1)
    #         temp_label = torch.argmax(temp_prob)
    #         # Success
    #         if temp_label != orig_label:
    #             output.num_changes += 1
    #             final_words[top_index[0]] = substitute
    #             output.changes.append([keys[top_index[0]][0], substitute, tgt_word])
    #             output.final_text = temp_text
    #             output.success_indication = 'Attack success'

    #             return output
    #         else:

    #             label_prob = temp_prob[orig_label]
    #             gap = current_prob - label_prob
    #             if gap > most_gap:
    #                 most_gap = gap
    #                 candidate = substitute

    #     if most_gap > 0:
    #         feature.num_changes += 1
    #         feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
    #         current_prob = current_prob - most_gap
    #         final_words[top_index[0]] = candidate

    # feature.final_text = tokenizer.convert_tokens_to_string(final_words)
    # feature.success_indication = 'Attack fail'

    print(
        output.num_changes,
        output.changes,
        output.query_length,
        output.success_indication,
    )

    return output


# @staticmethod
def add_specific_args(
    parser: argparse.ArgumentParser, root_dir: str
) -> argparse.ArgumentParser:
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--top-k", default=32, type=int)
    parser.add_argument("--change_ratio_limit", default=0.5, type=float)

    return parser

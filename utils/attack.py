import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from utils.dataprocessor import OutputFeatures

def get_sim_embed(embed_path, sim_path):
    id2word = {}
    word2id = {}

    with open(embed_path, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in id2word:
                id2word[len(id2word)] = word
                word2id[word] = len(id2word) - 1

    cos_sim = np.load(sim_path)
    return cos_sim, word2id, id2word


def get_important_scores(
    processor,
    target_features,
    tgt_model,
    current_prob,
    pred_label,
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
                :, pred_label
            ]  # Difference between original logit output and 1 masked logit output
            + (  # Add score which In case the results change.
                leave_1_probs_argmax != pred_label
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

  
def replacement_using_BERT(feature, current_prob, output,pred_label, word_index_with_I_score, processor, word_pred_idx, word_pred_scores_all, cos_mat = None, w2i ={},i2w={} ,threshold_pred_score = 3.0):
    
    final_words = copy.deepcopy(feature.input_ids) # tokenized word ids include CLS, SEP 

    for top_index , important_score in word_index_with_I_score:
        # limit ratio of word change
        if output.num_changes > int(args.change_ratio_limit * (len(final_words)-2)):
            output.success_indication = 'exceed change ratio limit'  # exceed
            return output

        tgt_word = feature.input_ids[top_index+1] #because of CLS, need to +1 

        #no filter words #####
        #if tgt_word in filter_words:
        #    continue
        ############################
        ##  Maybe useless ##########
        ############################
        if top_index > args.max_seq_length - 2:
            continue
        ############
        
        substitutes = word_pred_idx[top_index].unsqueeze(0)  # L, k
        word_pred_scores = word_pred_scores_all[top_index].unsqueeze(0)

        ###############################
        ####  Can be depreciated  #####
        ###############################
        substitutes = get_substitues(
            substitutes,
            word_pred_scores,
            threshold_pred_score,
        )
        ################################
        most_gap = 0.0
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_

            if substitute == tgt_word:
                continue  # filter out original word
            '''        
            if '##' in substitute:
                continue  # filter out sub-word
            '''
            if substitute in filter_words:
                continue
            '''
            if substitute in w2i and tgt_word in w2i:
                if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                    continue
            '''

            #Replace word and check whether the attack is successful
            temp_replace = final_words
            temp_replace[top_index+1] = substitute # replace token


            input_tensor = processor.get_tensor(temp_replace).to('cuda')    

            with torch.no_grad():       
                logit = finetuned_model(input_tensor)[0] 
            temp_logit= logit.detach() 
            temp_prob = torch.softmax(temp_logit, -1) 
            temp_label = torch.argmax(temp_logit, dim=1).flatten()  

            feature.query += 1

            #Success
            if temp_label != pred_label:
                output.num_changes += 1
              
                ####### ids_to_token & tokens_to_string######
                ########  may be converted to function ######
                substitute_token = processor.tokenizer.convert_ids_to_token(substitute)
                tgt_word_token = processor.tokenizer.convert_ids_to_token(tgt_word)
                output.changes.append([top_index, substitute_token, tgt_word_token])
                #############################################
                temp_replace_token = processor.tokenizer.convert_ids_to_token(temp_replace)
                temp_text = processor.tokenizer.convert_tokens_to_string(temp_replace_token)
                ##############################################

                output.final_text = temp_text
                output.success_indication = "Attack success"
                
                return
            else:

                label_prob = temp_prob[pred_label]
                gap = current_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            output.num_changes += 1

            ####### ids_to_token & tokens_to_string######
            candidate_token = processor.tokenizer.convert_ids_to_token(candidate)
            tgt_word_token = processor.tokenizer.convert_ids_to_token(tgt_word)
            output.changes.append([top_index, candidate_token, tgt_word_token])
            #############################################

            current_prob = current_prob - most_gap
            final_words[top_index+1] = candidate

    final_words_token = processor.tokenizer.convert_ids_to_token(final_words)
    final_text = processor.tokenizer.convert_tokens_to_string(final_words_token)
    output.final_adverse = final_text
    output.success_indication = 'Attack fail'
    return  


def get_substitues(substitutes, substitutes_score=None, threshold=3.0):
    # substitues L,k
    words = []
    sub_len, k = substitutes.size() #sub_len : # of subwords
    # Empty list (no substitution)
    if sub_len == 0:
        return words
    # Single word, choose word which score of substitutes is higher than threshold   
    for (substitute , I_score) in zip(substitutes[0], substitutes_score[0]):
        if threshold != 0 and I_score < threshold:
            break
        words.append(substitute)
    # return the ids that I_score exceed the threshold
    return words

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
    orig_label = torch.argmax(torch.tensor(feature.label_id))
    current_prob = pred_logit.max()

    if pred_label != orig_label:
        output.success_indication = "Predict fail"
        return output

    # word prediction은 MLM 모델에서 각 토큰 당 예측 값을 뽑음
    word_predictions = word_predictions[1:-1, :]  # except  [CLS], [SEP]
    # Top-K 개를 뽑아서 가장 높은 스코어 순으로 정렬하며, 가장 plausible 한 예측값들의 모음
    # torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
    word_pred_scores_all, word_pred_idx = torch.topk(
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

    ##########################################
    # important_score 다음 프로세스 (TBD)#########
    ##########################################
    # legacy code
    output.query_length += int(len(words))
    # sort by important score
    word_index_with_I_score= sorted(
        enumerate(important_scores), key=lambda x: x[1], reverse=True
        )  # sort the important score and index
    # print(list_of_index)
    #=> [(59, 0.00014871359), (58, 0.00011396408), (60, 0.00010085106), .... ]      [(index, Importacne score), ....]

    print('loading sim-embed')
    
    if args.use_sim_mat == 1:
        cos_mat, w2i, i2w = get_sim_embed('data/target_data/counter-fitted-vectors.txt', 'data/target_data/cos_sim_counter_fitting.npy')
    else:        
        cos_mat, w2i, i2w = None, {}, {}

    print('finish get-sim-embed')

    replacement_using_BERT(feature, 
                           current_prob, 
                           output,pred_label, 
                           word_index_with_I_score, 
                           processor, 
                           word_pred_idx, 
                           word_pred_scores_all,
                           cos_mat,
                           w2i,
                           i2w, 
                           args.threshold_pred_score)
    

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
    parser.add_argument("--threshold-pred-score", default=0.1, type=float)
    parser.add_argument("--use_sim_mat", type=int, help='whether use cosine_similarity to filter out atonyms')
    return parser

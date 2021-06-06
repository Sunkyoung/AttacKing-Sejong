import argparse

import torch
from torch.utils.data import DataLoader, SequentialSampler

from utils.dataprocessor import OutputFeatures

def get_important_scores(
    words,
    tokenizer,
    target_features,
    tgt_model,
    current_prob,
    orig_label,
    pred_logit,
    batch_size,
    max_length
):
    masked_features = get_masked_by_token(words)
    texts = [' '.join(words) for words in masked_words]
    all_input_ids = []
    all_masks = [] 
    all_segs = []

    for text in texts: 
        inputs = tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=max_length, )
        inputs_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        padding_length = max_length - len(input_ids)

        input_ids = input_ids + (padding_length * [0])
        token_type_ids = token_type_ids + (padding_length * [0])
        attention_mask = attention_mask +(padding_length * [0])

        all_input_ids.append(input_ids)
        all_segs.append(token_type_ids)
        all_masks.append(attention_mask)

    seqs = torch.tensor(all_input_ids, dtype=torch.long)
    segs = torch.tensor(all_segs, dtype=torch.long)
    masks = torch.tensor(all_masks, dtype=torch.long)

    seqs = seqs.to('cuda')

    eval_data = TensorDataset(seqs)
    # Run prediction for full data 
    eval_sampler = SequentialSampler(eval_data)        
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=batch_size
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

def get_masked_by_token(self, words):
    len_text = len(words)
    masked_words = [] 
    for i in range(len_text):
        masked_words.append(words[0:i]+['[MASK]']+words[i+1:])
    return masked_words 

def replacement_using_BERT(feature, 
                           words, 
                           keys, 
                           args,
                           pretrained_model, 
                           current_prob, 
                           output, 
                           pred_label, 
                           word_index_with_I_score, 
                           tokenizer, 
                           word_pred_idx, 
                           word_pred_scores_all, 
                           threshold_pred_score = 3.0):
    
    final_words = copy.deepcopy(words) # tokenized word ids include CLS, SEP 

    for top_index , important_score in word_index_with_I_score:
        # limit ratio of word change
        if output.num_changes > int(args.change_ratio_limit * (len(words))):
            output.success_indication = 'exceed change ratio limit'  # exceed
            return 

        tgt_word = words[top_index] #because of CLS, need to +1 

        #no filter words #####
        #if tgt_word in filter_words:
        #    continue
        ############################
        ##  need          ##########
        ############################
        if keys[top_index][0] > args.max_seq_length - 2:
            continue
        ############################
        tgt_word_sub_idx_start = keys[top_index][0]
        tgt_word_sub_idx_end = keys[top_index][1]

        substitutes = word_pred_idx[tgt_word_sub_idx_start : tgt_word_sub_idx_end]
        word_pred_scores = word_pred_scores_all[tgt_word_sub_idx_start : tgt_word_sub_idx_end]

        ###############################
        ####  Can be depreciated  #####
        ###############################
        substitutes = get_substitues(
            substitutes,
            tokenizer,
            args.use_bpe,
            pretrained_model,
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
            temp_replace[top_index] = substitute # replace token
            temp_text = tokenizer.convert_tokens_to_string(temp_replace)
            inputs = tokenizer.encode_plus(temp_text,None, add_special_tokens=True,max_length = args.max_seq_length,)
            inputs_ids = torch.tensor(inputs["inputs_ids"]).unsqueeze(0).to('cuda')
            seq_len = input_ids.size(1)

            

            with torch.no_grad():       
                logit = finetuned_model(input_ids)[0] 
            temp_logit= logit.detach() 
            temp_prob = torch.softmax(temp_logit, -1) 
            temp_label = torch.argmax(temp_logit, dim=1).flatten()  

            feature.query += 1

            #Success
            if temp_label != pred_label:
                output.num_changes += 1
                
                final_words[top_index] = substitute
                output.changes.append([keys[top_index][0], substitute, tgt_word])
                output.final_text = temp_text
                output.success_indication ="Attack success"
                return
            else:

                label_prob = temp_prob[pred_label]
                gap = current_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            output.num_changes +=1
            output.changes.append([keys[top_index][0], substitute, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index] = candidate

    
    final_text = tokenizer.convert_tokens_to_string(final_words)
    output.final_adverse = final_text
    output.success_indication = 'Attack fail'
    return  


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    words = []
    sub_len, k = substitutes.size() #sub_len : # of subwords
    # Empty list (no substitution)
    if sub_len == 0:
        return words
    # Single word, choose word which score of substitutes is higher than threshold   
    elif sub_len == 1:
        for (substitute , I_score) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and I_score < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(substitute)))
    # return the ids that I_score exceed the threshold
    else: 
        if use_bpe == True :
            words = get_bpe_substitutes(substitutes, tokenizer, mlm_model)
        else:
            return words 
    return words 

def get_bpe_substitutes(substitutes, tokenizer, mlm_model):
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4] # (maximun subwords number, maximun subtitutes)

    # find all possible candidates 

    all_substitutes = []
    print('subsitutes')
    print(substitutes)
        
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            print('lev_i')
            print(lev_i)
            all_substitutes = [[int(c)] for c in lev_i]
            print('all_substitutes')
            print(all_substitutes)
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i
            print('######')
            print(' alll')
            print(all_substitutes)

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    ###????? why 24?
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words


def run_attack(args, processor, example, feature, pretrained_model, finetuned_model):
    output = OutputFeatures(label_id=example.label, first_seq=example.first_seq)
    input_tensor = processor.get_tensor(feature.input_ids).unsqueeze(0).to('cuda')
    input_mask_tensor = processor.get_tensor(feature.input_mask).unsqueeze(0).to('cuda')

    words, sub_words, keys = processor.get_keys(example)

    with torch.no_grad():
        logit = finetuned_model(
            input_tensor, token_type_ids=None, attention_mask=input_mask_tensor
        )[0]
        word_predictions = pretrained_model(input_tensor)[0].detach()

    pred_logit = logit.detach()  # orig prob -> pred logit 으로 변경
    pred_label = torch.argmax(
        pred_logit, dim=1
    ).flatten()  # orig label -> pred label 으로 변경
    current_prob = pred_logit.max()

    if pred_label != feature.label_id:
        output.success_indication = "Predict fail"
        return output


    #sub_words = ['[CLS]'] + sub_words[:args.max_seq_length-2] + ['[SEP]']

    # word prediction은 MLM 모델에서 각 토큰 당 예측 값을 뽑음
    word_predictions = word_predictions[1:-1, :]  # except  [CLS], [SEP]
    # Top-K 개를 뽑아서 가장 높은 스코어 순으로 정렬하며, 가장 plausible 한 예측값들의 모음
    # torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
    word_pred_scores_all, word_pred_idx = torch.topk(
        word_predictions, args.top_k, -1
    )  # seq-len k  #top k prediction

    important_score = get_important_scores(
        words,
        processor.tokenizer,
        feature,
        finetuned_model,
        current_prob,
        pred_label,
        pred_logit,
        args.batch_size,
        args.max_length
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
    
    replacement_using_BERT(feature,
                           words,
                           keys,
                           args,
                           pretrained_model, 
                           current_prob, 
                           output,
                           pred_label, 
                           word_index_with_I_score, 
                           processor.tokenizer, 
                           word_pred_idx, 
                           word_pred_scores_all, 
                           args.threshold_pred_score)
    

    print(
        output.num_changes,
        output.changes,
        output.query_length,
        output.success_indication,
    )

    return output


@staticmethod
def add_specific_args(
    parser: argparse.ArgumentParser, root_dir: str
) -> argparse.ArgumentParser:
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--top-k", default=32, type=int)
    parser.add_argument("--change_ratio_limit", default=0.5, type=float)
    parser.add_argument("--threshold-pred-score", default=0.1, type=float)
    return parser

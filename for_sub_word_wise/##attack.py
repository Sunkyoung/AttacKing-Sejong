def attack(feature, tgt_model, mlm_model, tokenizer, k, batch_size, max_length=512, cos_mat=None, w2i={}, i2w={}, use_bpe=1, threshold_pred_score=0.3):
    # MLM-process
    #words, sub_words, keys = _tokenize(feature.seq, tokenizer)

    # original label
    inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, )
    
    input_ids = torch.tensor(inputs["input_ids"])
    token_type_ids = torch.tensor(inputs["token_type_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"])
    
    #print('##@#2#@#@#@#@#@#@')
    #print(input_ids.shape, token_type_ids.shape, attention_mask.shape)
    sub_words = tokenizer.convert_ids_to_tokens(inputs["input_ids"]) #special tokens are included 


    seq_len = len(inputs["input_ids"])

    orig_probs = tgt_model(input_ids.unsqueeze(0).to('cuda'),
                           attention_mask.unsqueeze(0).to('cuda'),
                           token_type_ids.unsqueeze(0).to('cuda')
                           )[0].squeeze()
   
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs)
    current_prob = orig_probs.max()
    
    if orig_label != feature.label:
        feature.success = 3
        return feature

    input_ids_ = input_ids.unsqueeze(0)     
    #input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
    word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab.   
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k  #top k prediction 
    #############################################################
    #Maybe fault?? 
    #############################################################
    word_predictions = word_predictions[1:len(sub_words), :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words), :]
    ##################################################################

    #print('sub_word len')
    #print(len(sub_words)) #262
    #print(input_ids)
    important_scores = get_important_scores(input_ids, token_type_ids.tolist(), attention_mask.tolist(), tgt_model, current_prob, orig_label, orig_probs,  #get important score 
                                            tokenizer, batch_size, max_length)
    feature.query += int(len(sub_words))
    #sort by important score
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True) # sort the important score and index 
    
    #print('listof ')
    #print(len(list_of_index)) # 261 # except the last word 
    # print(list_of_index)  
    # => [(59, 0.00014871359), (58, 0.00011396408), (60, 0.00010085106), .... ]      [(index, Importacne score), ....]
    final_words = copy.deepcopy(sub_words) #whole word corpus from mlm 

     
    for top_index, important_score in list_of_index:
        #limit ratio of word change
        if feature.change > int(0.4 * (len(sub_words))): 
            # allow the word can be changed until epsilon percent (0.4) number of words 
            feature.success = 1  # exceed
            return feature

        tgt_word = sub_words[top_index]
        '''
        if tgt_word in filter_words:
            continue
        '''
        if top_index > max_length - 2:
            #let`s cut down words early 
            continue


        substitutes = word_predictions[top_index].unsqueeze(0)  # L, k
        word_pred_scores = word_pred_scores_all[top_index].unsqueeze(0)

        
        substitutes = get_substitues(substitutes, tokenizer, mlm_model, use_bpe, word_pred_scores, threshold_pred_score)

        print('substitutes')
        print(substitutes)

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
            inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length, )
            input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to('cuda')
            seq_len = input_ids.size(1)
            temp_prob = tgt_model(input_ids)[0].squeeze()
            feature.query += 1
            temp_prob = torch.softmax(temp_prob, -1)
            temp_label = torch.argmax(temp_prob)
            #Success
            if temp_label != orig_label:
                feature.change += 1
                final_words[top_index] = substitute
                feature.changes.append([top_index, substitute, tgt_word])
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
            feature.changes.append([top_index, candidate, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index] = candidate

    feature.final_adverse = (tokenizer.convert_tokens_to_string(final_words))
    feature.success = 2
    return feature
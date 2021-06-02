def get_important_scores(input_ids, token_type_ids, attention_mask, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    
    masked_input_ids = _get_masked(input_ids,tokenizer)

    all_input_ids = []
    all_masks = []
    all_segs = []

    padding_length = max_length - len(input_ids)
    token_type_ids = token_type_ids + (padding_length * [0])
    attention_mask = attention_mask + (padding_length * [0])
    # Put 0 padding to input for all texts
    for input_ids in masked_input_ids:
        
        input_ids = input_ids.tolist() + (padding_length * [0])
        
        
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
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    leave_1_probs = []

    for batch in eval_dataloader:
        #iteratively get prediction of masked sequence
        masked_input, = batch 
        bs = masked_input.size(0)

        leave_1_prob_batch = tgt_model(masked_input)[0]  # B num-label
        leave_1_probs.append(leave_1_prob_batch)

    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    
    import_scores = (orig_prob
                     - leave_1_probs[:, orig_label] #Difference between original logit output and 1 masked logit output
                     + # Add score which In case the results change.
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()

    return import_scores
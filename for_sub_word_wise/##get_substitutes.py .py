def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size() #sub_len : # of subwords

    # Empty list (no substitution)
    if sub_len == 0:
        return words
    
    # Single word, choose word which score of substitutes is higher than threshold   
    for (substitute , I_score) in zip(substitutes[0], substitutes_score[0]):
        if threshold != 0 and I_score < threshold:
            break
        words.append(tokenizer._convert_id_to_token(int(substitute)))
    # composed of more than 2 subword

    return words
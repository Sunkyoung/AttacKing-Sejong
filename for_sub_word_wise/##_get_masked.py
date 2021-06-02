
"""
parameter : words : list[str]
return :  list of word list which one of word is masked one by one except last token.

example : "나는 밥을 좋아한다." -> [['[MASK]', '##는', '밥', '##을', '좋아', '##한다'], [['나', [MASK], '밥', '##을', '좋아', '##한다']]
"""

def _get_masked(input_ids, tokenizer):
    len_text = len(input_ids)
    mask_id = tokenizer.encode(['[MASK]'],add_special_tokens=False)[0]
    #print('mask_id')
    #print(mask_id)
    masked_input_ids = []
    for i in range(1,len_text-1):
    	ith_word_masked = copy.deepcopy(input_ids)
    	ith_word_masked[i] = mask_id
    	masked_input_ids.append(ith_word_masked)
    
    #print('masked_input_ids')
    #print(masked_input_ids)
    return masked_input_ids
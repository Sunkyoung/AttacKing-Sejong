from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import re
import time
import argparse
from konlpy.tag import Mecab


"""
Use only korean and english
"""
def preprocess(news_raw_list):
    hangul = re.compile(r'[^a-z가-힣A-Z]')
    news_list = np.array([hangul.sub(' ', s) for s in news_raw_list])
    mecab = Mecab()
    p_news_list = []
    for news in news_list:
        p_news_list.append(' '.join([word_pair[0] for word_pair in mecab.pos(news)]))
    return np.array(p_news_list)

"""
Make word dictionary
"""
def bag_of_word(tokens_list):
    word_dict = {}
    for tokens in tokens_list:
        for token in tokens:
            if not token in word_dict:
                word_dict[token] = len(word_dict)
    return word_dict

"""
list(vector) to string list
"""
def vec2str(vec):
    str_vec = [str(element) for element in vec]
    output = ' '.join(str_vec)
    return output

def main(args):

    input_path = args.input_path
    model_path = args.model_path
    
    input_df = pd.read_csv(input_path, sep='\t')
    text_list = input_df['title']
    pr_text_list = preprocess(text_list)
    
    tokens_list = [sent.split() for sent in pr_text_list]
    word_dict = bag_of_word(tokens_list)
    word_list = list(word_dict.keys())
    #Train word2vec model
    if args.training:
        start = time.time()
        model = Word2Vec(sentences=tokens_list, size=500, window=5, min_count=2, workers=4, sg=0)
        print("Training time : ",time.time()-start)
        model.save(model_path)
        print("Model saved")
    else:
       model = Word2Vec.load(model_path)
       print('model_loaded')
    text_path = args.text_path
    model_path = args.model_path
    

    with open(text_path, 'w',encoding='utf-8-sig') as f:
        for word in word_list:
            if word in model.wv:
                row = word
                row += " "
                row += vec2str(model.wv[word])
                f.write(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("--input_path", require = True)
    parser = argparse.ArgumentParser("--model_path", require = True)
    parser = argparse.ArgumentParser("--text_path", default = "")
    parser = argparse.ArgumentParser("--training", default = "True")
    parser.add_argument
    args = parser.parse_args()
    main(args)
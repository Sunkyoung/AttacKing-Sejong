

# CS475 Final Project (AttacKing Sejong)

## Replication BERT-ATTACK (EMNLP 2020) in Korean [[PAPER]](https://arxiv.org/abs/2004.09984)
- The reason why our project name is 'AttacKing Sejong' 😂 
- Model : KLUE-bert-base
- Dataset : KLUE-TC (a.k.a YNAT), NSMC (TBD: KLUE-NLI)
<br>

## Experimental Results
- Reimplementation

| Dataset | Original Accuracy | Attack Accuracy | Perturbation % | Query Num |
| :---: | :---: | :---: | :---: | :---: |
| IMDB | 93.6 | 1.2 | 4.0 | 647.9 |

- Korean dataset

| Dataset | Original Accuracy | Attack Accuracy | Perturbation % | Query Num |
| :---: | :---: | :---: | :---: | :---: |
| KLUE-TC | 83.2 | 0 | 18.0 | 31.1 |
| NSMC | 89.3 | 43.6 | 12.7 | 15.0 |


## Scripts
- KLUE-TC (YNAT)
    ~~~
    python main.py --dataset YNAT --input-dir data/target_data/ynat --output-dir output --finetuned-model-path output/target_model_YNAT.pt --output-file attacked_result_YNAT.json --counter-fitted-vector-txt data/counter_fitted_vector/counter_fitted_vectors_nsmc.txt --counter-fitted-vector-npy data/counter_fitted_vector/cos_sim_counter_fitting_ynat.npy
    ~~~
- NSMC
    ~~~
    python main.py --dataset NSMC --input-dir data/target_data/nsmc --output-dir output --finetuned-model-path output/target_model_NSMC.pt --output-file attacked_result_NSMC.json --counter-fitted-vector-txt data/counter_fitted_vector/counter_fitted_vectors_nsmc.txt --counter-fitted-vector-npy data/counter_fitted_vector/cos_sim_counter_fitting_nsmc.npy
    ~~~
- If you want to train in original method (word-wise attack), add argument `--run-wordwise-legacy`
<br>

## Dependencies
- Python 3.8.5
- PyTorch 1.7.1
- transformers 4.6.1
<br>

## References
- [BERT-Attack](https://github.com/LinyangLee/BERT-Attack)
- [NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)
- [KLUE-TC (a.k.a YNAT)](https://klue-benchmark.com/tasks/66/overview/description)

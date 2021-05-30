import argparse
from data.dataprocessor import YnatProcessor
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

def main(args):
    model_name = "klue/bert-base"

    mlm_config = AutoConfig.from_pretrained(model_name)
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name, config=mlm_config)
    mlm_model.to('cuda')
   
    processor = YnatProcessor()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(processor.get_labels())
    target_examples = processor.get_target_examples('./data/target/ynat-v1')
    target_features = processor.convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)                 

    # all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    # all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    # all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    # train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # if args.local_rank == -1:
    #     train_sampler = RandomSampler(train_data)
    # else:
    #     train_sampler = DistributedSampler(train_data)
    # mlm_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    target_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    finetuned_model = AutoModel.from_pretrained(model_name)
    finetuned_model = AutoModelForMaskedLM.from_pretrained(model_name, config=target_config)
    finetuned_model.to('cuda')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",default=None,type=str,required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    ### kyoungman.bae @ 19-05-28 @ 
    parser.add_argument("--openapi_key", default=None, type=str, required=True,
                        help="The openapi accessKey. Please go to this site(http://aiopen.etri.re.kr/key_main.php).")
    parser.add_argument("--bert_model_path", default=None, type=str, required=True,
                        help="Bert pre-trained model path")
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    main(args)
    
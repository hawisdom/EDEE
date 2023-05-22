# coding=utf-8
import argparse
import logging
import random
import numpy as np
import torch
from datasets import load_datasets_and_vocabs
from models import EDEE
from trainer import train

logger = logging.getLogger()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_path', type=str, default='./data', help='Dataset path.')
    parser.add_argument('--dataset_name', type=str, default='ChFinAnn',help='Choose ChFinAnn dataset.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to store output data.')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Directory to store cache data.')
    parser.add_argument('--role_role_num', type=int, default=1013, help='Number of classes.')
    parser.add_argument('--seed', type=int, default=2022, help='random seed for initialization')
    parser.add_argument('--cuda_id', type=str, default='0', help='Choose which GPUs to run')

    # Model parameters
    parser.add_argument('--embedding_dir', type=str, default='./model', help='Directory storing embeddings')
    parser.add_argument('--word_embedding_dim', type=int, default=768, help='Dimension of embeddings')
    parser.add_argument('--word_type_embedding_dim', type=int, default=50, help='Dimension of word_type embeddings')

    # MLP
    parser.add_argument('--hidden_size', type=int, default=200,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers of bilstm.')
    parser.add_argument('--num_mlps', type=int, default=4, help='Number of mlps in the last of model.')
    parser.add_argument('--final_hidden_size', type=int, default=200, help='Hidden size of mlps.')

    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for embedding.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=15.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20,
                        help="Log every X updates steps.")


    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations
    '''
    logger.info(vars(args))


def main():
    # Setup logging
    for h in logger.handlers:
        logger.removeHandler(h)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)

    # Parse args
    args = parse_args()
    check_args(args)

    # Setup CUDA, GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    # Load datasets and vocabs
    train_dataset,train_labels_weight,test_dataset,test_labels_weight,word_vocab,wType_tag_vocab = load_datasets_and_vocabs(args)

    # Build Model
    model = EDEE(args,wType_tag_vocab['len'])
    model.to(args.device)

    # Train
    _, ef_results, er_results, eu_results, eo_results, ep_results = train(args,model,train_dataset,test_dataset,train_labels_weight,test_labels_weight)

    if len(ef_results):
        best_eval_result = max(ef_results, key=lambda x: x['f1'])
        for key in sorted(best_eval_result.keys()):
            logger.info("ef_best:  %s = %s", key, str(best_eval_result[key]))
    if len(er_results):
        best_eval_result = max(er_results, key=lambda x: x['f1'])
        for key in sorted(best_eval_result.keys()):
            logger.info("er_best  %s = %s", key, str(best_eval_result[key]))
    if len(eu_results):
        best_eval_result = max(eu_results, key=lambda x: x['f1'])
        for key in sorted(best_eval_result.keys()):
            logger.info("eu_best  %s = %s", key, str(best_eval_result[key]))
    if len(eo_results):
        best_eval_result = max(eo_results, key=lambda x: x['f1'])
        for key in sorted(best_eval_result.keys()):
            logger.info("eo_best  %s = %s", key, str(best_eval_result[key]))
    if len(ep_results):
        best_eval_result = max(ep_results, key=lambda x: x['f1'])
        for key in sorted(best_eval_result.keys()):
            logger.info("ep_best  %s = %s", key, str(best_eval_result[key]))


if __name__ == "__main__":
    main()


# -*- coding: utf-8 -*-

import sys, codecs
import argparse
import numpy as np

import torch
import transformers
from transformers import AutoTokenizer
from transformers import BertModel

import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')

    parser.add_argument('-m',
                        '--bert_model',
                        help='Language model to get word vectors',
                        default='bert-base-uncased')

    parser.add_argument('-c',
                        '--cased',
                        help='Use this to consider upper/lower case distinction',
                        action='store_true')

    parser.add_argument('-b', '--batch_size', default=32, type=int)


    parser.add_argument('-s', '--start_layer', default=12, type=int)

    parser.add_argument('-e', '--end_layer', type=int)

    parser.add_argument('-f',
                        '--freq_threshold',
                        help='Words whose frequency is more than this value is considered',
                        default=10, type=int)

 
    args = parser.parse_args()

    return args


"""
To calculate mean norms of all tokens in the given corpus with their
frequencies.
"""
def cal_kappa_with_freqs(vectorizer,
                         tokenizer,
                         sentences,
                         start_layer,
                         end_layer,
                         freq_threshold=10,
                         is_split_into_words=True,
                         device='cpu'):

    token2sum_vec, token2freq = util.tokenize_and_sum_vectorize(vectorizer,
                                                                tokenizer,
                                                                sentences,
                                                                start_layer=start_layer,
                                                                end_layer=end_layer,
                                                                device=device)

    token2kappa_freq = {}
    vector_size = None
    for token, sum_vec in token2sum_vec.items():
        freq = token2freq[token]
        if freq >= freq_threshold:
            vector_size = sum_vec.size
            mean_norm = np.linalg.norm(sum_vec)/float(freq)
            kappa = util.cal_concentration(mean_norm, vector_size=vector_size)
            token2kappa_freq[token] = (kappa, freq)

    return token2kappa_freq


def output(token2kappa_freq, delim='\t', digit=3):


    for token, (kappa, freq) in token2kappa_freq.items():
        # to exclude middle and end subwords
        kappa = round(kappa, digit)
        output = delim.join((token, str(kappa), str(freq)))

        print(output)


def main():
    """
    This program detects words having wider meanings in the input source
    corpus than in the input target corpus.

        usage: python detect_meaning_differences.py SOURCE_CORPUS TARGET_CORPUS
    
    See 'def parse_args()' for other possible options.
    """

    args = parse_args()

    if not args.end_layer:
        args.end_layer = args.start_layer + 1

    # Preparing data
    sentences = util.load_sentences(args.corpus, args.cased)
    batched_sentences = util.to_batches(sentences, batch_size=args.batch_size)


    # Preparing BERT model and tokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vectorizer = BertModel.from_pretrained(args.bert_model, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Calculating mean vectors with token frequencies
    token2kappa_freq = cal_kappa_with_freqs(vectorizer,
                                            tokenizer,
                                            batched_sentences,
                                            start_layer=args.start_layer,
                                            end_layer=args.end_layer,
                                            freq_threshold=args.freq_threshold,
                                            device=device)

    output(token2kappa_freq)


if __name__ == '__main__':
    main()

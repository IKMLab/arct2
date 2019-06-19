"""Common manipulations of the ARCT data.

Many functions and classes copied from here:
https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py
"""
import os
import json
import numpy as np
import pandas as pd
import glovar


def glove():
    glove_path = os.path.join(glovar.ARCT_DIR, 'glove.npy')
    return np.load(glove_path)


def vocab():
    vocab_path = os.path.join(glovar.ARCT_DIR, 'vocab.json')
    with open(vocab_path, 'r') as f:
        return json.loads(f.read())


def rev_vocab():
    rev_vocab_path = os.path.join(glovar.ARCT_DIR, 'rev_vocab.json')
    with open(rev_vocab_path, 'r') as f:
        return json.loads(f.read())


def load(dataset):
    path = os.path.join(glovar.ARCT_DIR, '%s-full.txt' % dataset)
    return pd.read_csv(path, delimiter='\t')

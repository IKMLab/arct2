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


def make_adversarial(dataset):
    claim_map = pd.read_csv('adversarial_dataset/claim_map.csv')
    claim_map = dict(zip(claim_map.original.values, claim_map.negated.values))

    def mark_id_adversarial(x):
        return f'{x}-adversarial'

    original = load(dataset)

    new_claims = [claim_map[c] for c in list(original.claim)]
    new_labels = [not l for l in original.correctLabelW0orW1]

    adversarial = original.copy()
    adversarial['#id'] = adversarial['#id'].apply(mark_id_adversarial)
    adversarial.claim = new_claims
    adversarial.correctLabelW0orW1 = new_labels

    original['adversarial'] = [False] * len(original)
    adversarial['adversarial'] = [True] * len(adversarial)

    new_dataset = pd.concat([original, adversarial])

    return new_dataset


def view(x):
    print('-' * 8)
    print(f'Claim:    {x.claim}')
    print(f'Reason:   {x.reason}')
    print(f'Warrant0: {x.warrant0}')
    print(f'Warrant1: {x.warrant1}')
    print(f'Label:    {x.correctLabelW0orW1}')

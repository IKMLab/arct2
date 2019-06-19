"""Build vocab for ARCT data."""
import os
import json
import glovar
from arct import data
from util import text


def flatten(list_of_lists):
    return [x for sublist in list_of_lists for x in sublist]


if __name__ == '__main__':
    print('Building ARCT vocab...')

    # grab all sents from all data subsets
    datasets = ['train', 'dev', 'test']
    sent_cols = ['claim', 'reason', 'warrant0', 'warrant1']
    sents = []
    for dataset in datasets:
        df = data.load(dataset)
        for _, row in df.iterrows():
            for col in sent_cols:
                sents.append(row[col])

    # tokenize
    tokens = set(flatten([text.tokenize(s) for s in sents]))

    # build the vocab dictionary
    vocab = dict(zip(tokens, range(len(tokens))))
    rev_vocab = {v: k for k, v in vocab.items()}

    # save the vocab dictionary
    vocab_path = os.path.join(glovar.ARCT_DIR, 'vocab.json')
    rev_vocab_path = os.path.join(glovar.ARCT_DIR, 'rev_vocab.json')
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))
    with open(rev_vocab_path, 'w') as f:
        f.write(json.dumps(rev_vocab))

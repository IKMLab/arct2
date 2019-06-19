"""Build GloVe vectors for ARCT data."""
import os
import numpy as np
from arct import data
import glovar
from util import text


if __name__ == '__main__':
    print('Creating GloVe embeddings...')
    vocab = data.vocab()
    embeddings = text.create_glove_embeddings(vocab)
    save_path = os.path.join(glovar.ARCT_DIR, 'glove.npy')
    np.save(save_path, embeddings)

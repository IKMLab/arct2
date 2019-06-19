"""Text utilities."""
import re
import nltk
import numpy as np
import glovar


# regex to remove all Non-Alpha Numeric and space
special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)


# regex to replace all numeric
replace_numbers = re.compile(r'\d+', re.IGNORECASE)


def clean(text):
    # Clean the text, with the option to remove stopwords and to stem words.
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"i’m", "i am", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = replace_numbers.sub('', text)
    text = special_character_removal.sub('', text)
    return text.strip()


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [clean(t) for t in tokens]
    return tokens


def create_glove_embeddings(vocab):
    embeddings = np.random.normal(size=(len(vocab), 300))\
        .astype('float32', copy=False)
    with open(glovar.GLOVE_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split()
            if len(s) > 301:  # a hack I have seemed to require for GloVe 840B
                s = [s[0]] + s[-300:]
                assert len(s) == 301
            if s[0] in vocab.keys():
                embeddings[vocab[s[0]], :] = np.asarray(s[1:])
    return embeddings

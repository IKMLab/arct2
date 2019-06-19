#!/usr/bin/env bash

# make the data dir and subdirs
if [ ! -d data ]; then
    echo "Making data dir..."
    mkdir data
fi
if [ ! -d data/ckpts ]; then
    echo "Making ckpts dir..."
    mkdir data/ckpts
fi
if [ ! -d results ]; then
    echo "Making results dir..."
    mkdir results
fi
if [ ! -d data/arct ]; then
    echo "Making arct dir..."
    mkdir data/arct
fi
if [ ! -d data/glove ]; then
    echo "Making glove dir..."
    mkdir data/glove
fi

# download the data
echo "Downloading data..."
if [ ! -f data/arct/train-full.txt ]; then
    wget https://github.com/habernal/semeval2018-task12/raw/master/data/train/train-full.txt -P data/arct/
fi
if [ ! -f data/arct/dev-full.txt ]; then
    wget https://github.com/habernal/semeval2018-task12/raw/master/data/dev/dev-full.txt -P data/arct/
fi
if [ ! -f data/arct/test-only-data.txt ]; then
    wget https://github.com/habernal/semeval2018-task12/raw/master/data/test/test-only-data.txt -P data/arct/
fi
if [ ! -f data/arct/truth.txt ]; then
    wget https://github.com/habernal/semeval2018-task12-results/raw/master/data/gold/truth.txt -P data/arct/
fi

# merge the test labels
echo "Merging test labels..."
python merge_test_labels.py

# delete redundant files
echo "Cleaning up redundant files..."
rm data/arct/test-only-data.txt
rm data/arct/truth.txt

# download GloVe
if [ ! -f data/glove/glove.840B.300d.txt ]; then
    if [ ! -f data/glove/glove.840B.300d.zip ]; then
        echo "Downloading GloVe..."
        wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P data/glove/
    fi
    echo "Unzipping Glove..."
    unzip data/glove/glove.840B.300d.zip -d data/glove/ || exit 1
fi

# build the vocab and GloVe matrix for BOV experiments
python build_vocab.py
python build_glove.py

# clean up redundant GloVe files (ARCT GloVe vectors live in the data/arct dir)
echo "Deleting redundant GloVe files..."
rm -rf data/glove

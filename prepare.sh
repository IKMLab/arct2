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
if [ ! -f data/arct/train-full.txt ]; then
    echo "Downloading train data..."
    wget https://github.com/habernal/semeval2018-task12/raw/master/data/train/train-full.txt -P data/arct/
fi
if [ ! -f data/arct/dev-full.txt ]; then
    echo "Downloading dev data..."
    wget https://github.com/habernal/semeval2018-task12/raw/master/data/dev/dev-full.txt -P data/arct/
fi
if [ ! -f data/arct/test-full.txt ]; then
    # download the test set
    echo "Downloading test data..."
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
fi

# copy the adversarial dataset into data/arct
if [ ! -f data/arct/train-adv-full.txt ]; then
    cp adversarial_dataset/train.csv data/arct/train-adv-full.txt
fi
if [ ! -f data/arct/dev-adv-full.txt ]; then
    cp adversarial_dataset/dev.csv data/arct/dev-adv-full.txt
fi
if [ ! -f data/arct/test-adv-full.txt ]; then
    cp adversarial_dataset/test.csv data/arct/test-adv-full.txt
fi

# download GloVe
if [ ! -f data/arct/glove.npy ]; then
    if [ ! -f data/glove/glove.840B.300d.txt ]; then
        if [ ! -f data/glove/glove.840B.300d.zip ]; then
            echo "Downloading GloVe..."
            wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P data/glove/
        fi
        echo "Unzipping Glove..."
        unzip data/glove/glove.840B.300d.zip -d data/glove/ || exit 1
    fi
fi

# build the vocab and GloVe matrix for BOV experiments
if [ ! -f data/arct/vocab.json ]; then
    python build_vocab.py
fi
if [ ! -f data/arct/glove.npy ]; then
    python build_glove.py
fi

# clean up redundant GloVe files (ARCT GloVe vectors live in the data/arct dir)
if [ -f data/glove ]; then
    echo "Deleting redundant GloVe files..."
    rm -rf data/glove
fi

echo "Success."

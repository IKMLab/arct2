# Probing Neural Network Understanding of Natural Language Arguments

[Link](https://www.aclweb.org/anthology/P19-1459)

Authors: Timothy Niven and Hung-Yu Kao

Abstract:

> We are surprised to find that BERT's peak performance of 77\% on the Argument Reasoning Comprehension Task reaches just three points below the average untrained human baseline. However, we show that this result is entirely accounted for by exploitation of spurious statistical cues in the dataset. We analyze the nature of these cues and demonstrate that a range of models all exploit them. This analysis informs the construction of an adversarial dataset on which all models achieve random accuracy. Our adversarial dataset provides a more robust assessment of argument comprehension and should be adopted as the standard in future work.

Reference:

```
@inproceedings{niven-kao-2019-probing,
    title = "Probing Neural Network Comprehension of Natural Language Arguments",
    author = "Niven, Timothy  and
      Kao, Hung-Yu",
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1459",
    pages = "4658--4664",
    abstract = "We are surprised to find that BERT{'}s peak performance of 77{\%} on the Argument Reasoning Comprehension Task reaches just three points below the average untrained human baseline. However, we show that this result is entirely accounted for by exploitation of spurious statistical cues in the dataset. We analyze the nature of these cues and demonstrate that a range of models all exploit them. This analysis informs the construction of an adversarial dataset on which all models achieve random accuracy. Our adversarial dataset provides a more robust assessment of argument comprehension and should be adopted as the standard in future work.",
}
```

## Errata

Due to an errata in the original paper, we have updated the arXiv
version [here](https://arxiv.org/abs/1907.07355). The errata was that
our adversarial experiments used the original (not claim-negated) data
for training, augmented with swapped warrants. This is important for 
breaking the validity of spurious cues over the claims and warrants. The
negated claim augmentation has successfully eliminated warrant cues as
a solution. Details of the problem of additional cues over claims and 
warrants will be provided in a forthcoming paper. 

## Adversarial Dataset

Provided in the `adversarial_dataset` folder. The setup we used in the
experiments is
  * `train-swapped.csv`
  * `dev-adv-negated.csv`
  * `test-adv-negated.csv`

## Viewing our Results

Each experiment has its own folder in the `results_from_paper` folder.
The suffixes (and combinations thereof) indicate the setup
- `cw` only considers claims and warrants
- `rw` only considers reasons and warrants
- `w` only considers warrants
- `adv` uses the adversarial dev and test dataset, and swap augmented
  train dataset
The tables below also show the experiment names for reproducing specific
results.

Within each experiment's folder in `results` you will find
- `accs.csv`: contains accuracies for train, dev, and test over
  all random seeds
- `best_params.json`: lists the best parameters from grid search
- `grid.csv`: lists all grid search results and parameter
  combinations
- `preds.csv`: lists all predictions for all data points which
  can be filtered by dataset and queried by each data point's 
  unique identifier

You can get a summary of the accuracies over various random
seeds for an experiment by running

```
python accs.py experiment_name --from_paper
```

For details of how each experiment is run, you can view the
files in the `experiments` folder.

The `from_paper` argument will pull results from the 
`results_from_paper` directory. Without this flag it will read the 
`results` directory, where any experiment results you run yourself 
locally will be recorded.

## Reproducing our Results

### Virtual Environment

Package requirements are listed in `requirements.txt`. We used and 
tested this repository with Python 3.6. To simplify this, here are the  
commands I issued on my Ubuntu computer to make this repository work:
- `conda create --name arct2 python=3.6`
- `conda activate arct2`
- `pip install pandas==0.23.4`
- `pip install nltk==3.4`
- `pip install tqdm==4.28.1`
- `conda install -c pytorch pytorch=0.4.1`
- `pip install numpy==1.15.4`  (not sure if this is necessary, 
  but it does work)
- `pip install pytorch-pretrained-bert==0.1.2`

### Download and Prepare Data

Run `prepare.sh`. If you download a new version of this repository,
you should run this again.

### Running Experiments

Then to reproduce the results of any of the experiments run the
script

```
python run.py experiment_name
```

Note that due to a bug in the random seed control in the original 
experiments the exact numbers from the paper cannot be reproduced
(the data loaders were initialized before the seed was set). Having 
fixed the bug we are reproducing the experiments and reporting the 
exact values you should see when you run this code. As expected they
are slightly different, but not qualitatively so. Apparently the 
order in which the examples are presented does have some effect. BERT
Base has so far been able to get a lucky run up to 72.5\% on the 
original dataset.

### Results

#### Table 1

|Model                                  |Experiment Name|Dev (Mean)    |Test (Mean)   |Test (Median)|Test (Max)|
|---------------------------------------|---------------|--------------|--------------|-------------|----------|
|Human (trained)                        |               |              |0.909 +/- 0.11|             |          |
|Human (untrained)                      |               |              |0.798 +/- 0.16|             |          |
|BERT (Large)                           |bert_large     |0.694 +/- 0.04|0.660 +/- 0.08|0.690        |0.775     |
|GIST (Choi and Lee, 2018)              |               |0.716 +/- 0.01|0.711 +/- 0.01|             |          |
|BERT (Base)                            |bert_base      |0.675 +/- 0.03|0.634 +/- 0.07|0.661        |0.725     |
|World Knowledge (Botschen et al., 2018)|               |0.674 +/- 0.01|0.568 +/- 0.03|             |0.610     |
|Bag of Vectors (BoV)                   |bov            |0.633 +/- 0.02|0.564 +/- 0.02|0.562        |0.604     |
|BiLSTM                                 |bilstm         |0.659 +/- 0.01|0.544 +/- 0.02|0.547        |0.583     |

#### Table 4

These experiments are named following the convention 
`{experiment_name}_adv_orig`.

|Model        |Experiment Name|Adversarial Test (Mean)|Adversarial Test (Median)|Adversarial Test (Max)|
|-------------|---------------|-----------------------|-------------------------|----------------------|
|BERT (Large) |bert_large_adv |0.509 +/- 0.02         |0.509                    |0.539                 |
|BoV          |bov_adv        |0.500 +/- 0.00         |0.500                    |0.503                 |
|BiLSTM       |bilstm_adv     |0.499 +/- 0.00         |0.500                    |0.501                 |

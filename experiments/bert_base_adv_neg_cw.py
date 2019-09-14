"""BERT Base on the adversarial 2.0 dataset, just on claims and warrants.

This is the dataset where the training set includes negations. This however does
introduce strong spurious cues in the claim.

We also expand the grid space.
"""
from arct import bert
from util import training, experiments


def run():
    args = training.Args(
        experiment_name=__name__.split('.')[-1],
        bert_model='bert-base-uncased',
        max_seq_length=80,
        annealing_factor=0.1,
        num_train_epochs=50)
    grid_space = {
        'learning_rate': [9e-6, 6e-6, 3e-6, 1e-6],
        'dropout_prob': [.1, .4]}
    experiments.run(
        args=args,
        model_constructor=bert.BERT.from_args,
        data_loaders_constructor=bert.DataLoadersAdv2CW,
        grid_space=grid_space,
        n_experiments=20)

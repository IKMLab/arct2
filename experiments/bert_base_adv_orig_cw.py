"""Experiment for running ARCT through BERT base."""
from arct import bert
from util import training, experiments


def run():
    args = training.Args(
        experiment_name=__name__.split('.')[-1],
        bert_model='bert-base-uncased',
        max_seq_length=80,
        annealing_factor=0.1,
        learning_rate=8e-05,
        num_train_epochs=30)
    grid_space = {
        'learning_rate': [1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5]}
    experiments.run(
        args=args,
        model_constructor=bert.BERT.from_args,
        data_loaders_constructor=bert.DataLoadersAdvNegatedCW,
        grid_space=grid_space,
        n_experiments=20)

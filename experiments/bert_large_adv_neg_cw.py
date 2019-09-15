"""Experiment for BERT large on the doubled swap training set."""
from arct import bert
from util import training, experiments


def run():
    args = training.Args(
        experiment_name=__name__.split('.')[-1],
        bert_model='bert-large-uncased',
        annealing_factor=0.1,
        num_train_epochs=20,
        max_seq_length=80,
        learning_rate=2e-5)
    grid_space = {
        'learning_rate': [6e-5, 3e-5, 1e-5, 9e-6, 6e-6, 3e-6, 1e-6, 9e-7, 6e-7]}
    experiments.run(
        args=args,
        model_constructor=bert.BERT.from_args,
        data_loaders_constructor=bert.DataLoadersAdvNegatedCW,
        grid_space=grid_space,
        n_experiments=20)

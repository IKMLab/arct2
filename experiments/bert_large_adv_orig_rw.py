"""Experiment for running ARCT through BERT base."""
from arct import bert
from util import training, experiments


def run():
    args = training.Args(
        experiment_name=__name__.split('.')[-1],
        bert_model='bert-large-uncased',
        max_seq_length=80,
        num_train_epochs=20,
        annealing_factor=0.1)
    model_constructor = bert.BERT.from_args
    grid_space = {
       'learning_rate': [6e-5, 5e-5, 4e-5, 3e-5, 2e-5]}
    experiments.run(
        args=args,
        model_constructor=model_constructor,
        data_loaders_constructor=bert.DataLoadersAdvOriginalRW,
        grid_space=grid_space,
        n_experiments=20)

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
    model_constructor = bert.BERT.from_args
    experiments.run(
        args=args,
        model_constructor=model_constructor,
        data_loaders_constructor=bert.DataLoadersAdv2,
        grid_space=None,
        n_experiments=20,
        do_grid=False)

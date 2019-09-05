"""Experiment for BERT large adversarial, considering claims and warrants."""
from arct import bert
from util import training, experiments


def run():
    args = training.Args(
        experiment_name=__name__.split('.')[-1],
        bert_model='bert-large-uncased',
        annealing_factor=0.1,
        num_train_epochs=20,
        max_seq_length=60,
        learning_rate=2e-5)
    model_constructor = bert.BERT.from_args
    experiments.run(
        args=args,
        model_constructor=model_constructor,
        data_loaders_constructor=bert.DataLoadersAdvCW,
        grid_space=None,
        n_experiments=20,
        do_grid=False)

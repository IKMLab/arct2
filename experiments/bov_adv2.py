"""BOV model. Claims and Reasons are summed together - composed."""
from util import training, experiments
from arct import bov


def run():
    args = training.Args(
        experiment_name=__name__.split('.')[-1],
        use_bert=False,
        n_train_epochs=3,
        dropout_prob=0.,
        train_batch_size=32,
        tune_embeds=True)
    model_constructor = bov.BOV
    grid_space = {
        'learning_rate': [0.1, 0.09, 0.08],
        'n_train_epochs': [3, 5],
        'dropout_prob': [0., 0.1],
        'train_batch_size': [16, 32, 64]}
    experiments.run(
        args=args,
        model_constructor=model_constructor,
        data_loaders_constructor=bov.DataLoadersAdv2,
        grid_space=grid_space,
        n_experiments=20)

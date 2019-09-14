"""BOV model. Adversarial 2.0 dataset. Just claims and warrants."""
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
    grid_space = {
        'learning_rate': [0.1, .09, .06, .03, .01, .009, .006, .003, .001],
        'n_train_epochs': [3, 5, 10, 20, 30, 50],
        'dropout_prob': [0., .1, .2, .3, .4, .5],
        'train_batch_size': [16, 32, 64]}
    experiments.run(
        args=args,
        model_constructor=bov.BOV_W,
        data_loaders_constructor=bov.DataLoadersAdvNegated,
        grid_space=grid_space,
        n_experiments=20)

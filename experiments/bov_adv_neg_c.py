"""BOV model. Claims and Reasons are summed together - composed."""
from util import training, experiments
from arct import bov


def run():
    args = training.Args(
        experiment_name=__name__.split('.')[-1],
        use_bert=False,
        num_train_epochs=20,
        dropout_prob=0.,
        train_batch_size=32,
        tune_embeds=True)
    grid_space = {
        'learning_rate': [.03, .01, .009, .006, .003],
        'dropout_prob': [.1, .2, .3],
        'train_batch_size': [16, 32]}
    experiments.run(
        args=args,
        model_constructor=bov.BOV_C,
        data_loaders_constructor=bov.DataLoadersAdvNegated,
        grid_space=grid_space,
        n_experiments=20)

"""Baseline BiLSTM model."""
from util import training, experiments
from arct import bilstm


def run():
    args = training.Args(
        experiment_name=__name__.split('.')[-1],
        use_bert=False,
        tune_embeds=True,
        annealing_factor=0.1,
        num_train_epochs=30,
        train_batch_size=32,
        hidden_size=512,
        dropout_prob=0.1)
    model_constructor = bilstm.BiLSTM_CW
    grid_space = {
        'learning_rate': [.09, .06, .03, .01, .009, .006, .003, .001],
        'dropout_prob': [0., .1, .2, .3, .4, .5],
        'batch_size': [16, 32, 64],
        'hidden_size': [128, 256, 512]}
    experiments.run(
        args=args,
        model_constructor=model_constructor,
        data_loaders_constructor=bilstm.DataLoadersAdv2,
        grid_space=grid_space,
        n_experiments=20)

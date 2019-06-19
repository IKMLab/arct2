"""BiLSTM model that only considers the warrants."""
from util import training, experiments
from arct import bilstm


def run():
    args = training.Args(
        experiment_name=__name__.split('.')[-1],
        use_bert=False,
        tune_embeds=True,
        annealing_factor=0.1,
        num_train_epochs=20,
        hidden_size=512,
        train_batch_size=32,
        dropout_prob=0.1)
    model_constructor = bilstm.BiLSTM_WW
    data_loaders = bilstm.DataLoaders()
    grid_space = {
        'learning_rate': [0.08, 0.07, 0.06, 0.05]}
    experiments.run(
        args=args,
        model_constructor=model_constructor,
        data_loaders=data_loaders,
        grid_space=grid_space,
        n_experiments=20)

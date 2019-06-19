import os
import json
import unittest
import pandas as pd
from util import experiments, training
import glovar


class TestGrid(unittest.TestCase):

    def setUp(self):
        self.experiment_dir = os.path.join(glovar.RESULTS_DIR, 'test')
        self.grid_path = experiments.grid_path('test')
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)
        self.grid_search = experiments.GridSearch(
            'test', None, None, None, {'lr': [0.1, 0.3],
                                       'tune': [True, False]})

    # TODO: these are all broken

    def _test_winning_combinations_clear_winner(self):
        data = {
            'id': [0, 1, 2, 3],
            'lr': [0.1, 0.1, 0.3, 0.3],
            'tune': [True, False, True, False],
            'train_acc': [1, 1, 1, 1],
            'dev_acc': [1, 2, 3, 4],
            'test_acc': [1, 1, 1, 1]}
        df = pd.DataFrame(data=data, columns=data.keys())
        df.to_csv(self.grid_path, index=False)
        best_acc, combinations = self.grid_search.winning_combinations()
        self.assertEqual(4, best_acc)
        self.assertEqual(1, len(combinations))
        self.assertEqual({'id': 3, 'lr': 0.3, 'tune': False},
                         combinations[0])

    def _test_winning_combinations_no_clear_winner(self):
        data = {
            'id': [0, 1, 2, 3],
            'lr': [0.1, 0.1, 0.3, 0.3],
            'tune': [True, False, True, False],
            'train_acc': [1, 1, 1, 1],
            'dev_acc': [1, 2, 3, 3],
            'test_acc': [1, 1, 1, 1]}
        df = pd.DataFrame(data=data, columns=data.keys())
        df.to_csv(self.grid_path, index=False)
        best_acc, combinations = self.grid_search.winning_combinations()
        self.assertEqual(3, best_acc)
        self.assertEqual(2, len(combinations))
        self.assertEqual({'id': 2, 'lr': 0.3, 'tune': True}, combinations[0])
        self.assertEqual({'id': 3, 'lr': 0.3, 'tune': False}, combinations[1])

    def _test_winning_combinations_winner_on_tie_break(self):
        data = {
            'id': [0, 1, 2, 3, 0, 3],
            'lr': [0.1, 0.1, 0.3, 0.3, 0.1, 0.3],
            'tune': [True, False, True, False, True, False],
            'train_acc': [1, 1, 1, 1],
            'dev_acc': [3, 2, 2, 3, 2.6, 2.8],
            'test_acc': [1, 1, 1, 1]}
        df = pd.DataFrame(data=data, columns=data.keys())
        df.to_csv(self.grid_path, index=False)
        best_acc, combinations = self.grid_search.winning_combinations()
        self.assertEqual(3, best_acc)
        self.assertEqual(1, len(combinations))
        self.assertEqual({'id': 3, 'lr': 0.3, 'tune': False}, combinations[0])


class TestRunExperiments(unittest.TestCase):

    def setUp(self):
        self.experiment_dir = os.path.join(glovar.RESULTS_DIR, 'test')
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)
        self.params_path = experiments.params_path('test')
        self.accs_path = experiments.accs_path('test')
        self.preds_path = experiments.preds_path('test')
        if os.path.exists(self.accs_path):
            os.remove(self.accs_path)
        if os.path.exists(self.preds_path):
            os.remove(self.preds_path)
        with open(self.params_path, 'w') as f:
            f.write(json.dumps({'lr': 0.1, 'dropout': 0.1}))

    def test_run_with_no_existing_data(self):
        args = training.Args(experiment_name='test')
        model_constructor = lambda x: None
        data_loaders = None
        grid_space = None
        n_experiments = 3
        train_fn = FakeTrainFunction()
        experiments.run(args, model_constructor, data_loaders, grid_space,
                        n_experiments, train_fn)
        accs = pd.read_csv(self.accs_path)
        preds = pd.read_csv(self.preds_path)
        self.assertEqual(3, len(accs))
        self.assertEqual(18, len(preds))

    def test_run_with_existing_data(self):
        args = training.Args(experiment_name='test')
        model_constructor = lambda x: None
        data_loaders = None
        grid_space = None
        n_experiments = 3
        train_fn = FakeTrainFunction()
        experiments.run(args, model_constructor, data_loaders, grid_space,
                        n_experiments, train_fn)
        train_fn = FakeTrainFunction()
        # now we have some data, go again, resume onto 6
        n_experiments = 6
        experiments.run(args, model_constructor, data_loaders, grid_space,
                        n_experiments, train_fn)
        accs = pd.read_csv(self.accs_path)
        preds = pd.read_csv(self.preds_path)
        self.assertEqual(6, len(accs))
        self.assertEqual(36, len(preds))

    def tearDown(self):
        if os.path.exists(self.accs_path):
            os.remove(self.accs_path)
        if os.path.exists(self.preds_path):
            os.remove(self.preds_path)


class FakeTrainFunction:

    def __init__(self):
        self.i = 0
        self.accs = [
            {'train': 0.8, 'dev': 0.6, 'test': 0.5},
            {'train': 0.7, 'dev': 0.7, 'test': 0.6},
            {'train': 1.0, 'dev': 0.6, 'test': 0.4}]
        self.preds = [{
            'train': [
                {'id': 1, 'prob0': 0.2, 'prob1': 0.8, 'pred': 1, 'correct': 1},
                {'id': 2, 'prob0': 0.8, 'prob1': 0.2, 'pred': 0, 'correct': 0},
            ],
            'dev': [
                {'id': 1, 'prob0': 0.2, 'prob1': 0.8, 'pred': 1, 'correct': 1},
                {'id': 2, 'prob0': 0.8, 'prob1': 0.2, 'pred': 0, 'correct': 0},
            ],
            'test': [
                {'id': 1, 'prob0': 0.2, 'prob1': 0.8, 'pred': 1, 'correct': 1},
                {'id': 2, 'prob0': 0.8, 'prob1': 0.2, 'pred': 0, 'correct': 0},
            ]}] * 3

    def __call__(self, args, model, data_loaders):
        accs = self.accs[self.i]
        preds = self.preds[self.i]
        self.i += 1
        return accs, preds

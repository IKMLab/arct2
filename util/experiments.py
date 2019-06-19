"""Utilities for experiments."""
import os
import copy
import json
import random
import glovar
import itertools
import numpy as np
import pandas as pd
from util import training


def accs_path(experiment_name):
    return os.path.join(glovar.RESULTS_DIR, experiment_name, 'accs.csv')


def grid_path(experiment_name):
    return os.path.join(glovar.RESULTS_DIR, experiment_name, 'grid.csv')


def preds_path(experiment_name):
    return os.path.join(glovar.RESULTS_DIR, experiment_name, 'preds.csv')


def params_path(experiment_name):
    return os.path.join(
        glovar.RESULTS_DIR, experiment_name, 'best_params.json')


class GridSearch:

    def __init__(self, experiment_name, model_constructor, data_loaders, args,
                 search_space):
        self.experiment_name = experiment_name
        self.model_constructor = model_constructor
        self.data_loaders = data_loaders
        self.args = args
        self.search_space = search_space
        self.search_keys = search_space.keys()
        self.grid_path = grid_path(experiment_name)
        self.params_path = params_path(experiment_name)
        self.data, self.columns = self.get_or_load_data()

    def __call__(self):
        print('Conducting grid search for %s...' % self.experiment_name)

        for combination in self.combinations:
            if not self.evaluated(combination):
                self.evaluate(combination)
            else:
                print('Already evaluated this combination:')
                for key, value in combination.items():
                    print('\t%s:\t%s' % (key, value))

        best_acc, combinations = self.winning_combinations()
        while len(combinations) > 1:
            print('Found %s combinations with best acc of %s.'
                  % (len(combinations), best_acc))
            print('Performing tie break...')
            for _ in range(5):
                seed = random.choice(range(10000))
                for combination in combinations:
                    self.evaluate(combination, seed, tie_break=True)
            best_acc, combinations = self.winning_combinations()

        best_params = combinations[0]
        print('Grid search complete. Best acc: %s. Params:' % best_acc)
        print(best_params)

        print('Saving grid best params...')
        with open(self.params_path, 'w') as f:
            f.write(json.dumps(best_params))

        return best_acc, best_params

    @property
    def combinations(self):
        keys = self.search_space.keys()
        values = list(self.search_space.values())
        i = 0
        for _values in itertools.product(*values):
            combination = dict(zip(keys, _values))
            combination['id'] = i
            i += 1
            yield combination

    def evaluate(self, combination, seed=42, tie_break=False):
        print('Evaluating param combination%s:'
              % ' (tie break)' if tie_break else '')
        args = copy.deepcopy(self.args)
        for key, value in combination.items():
            setattr(args, key, value)
            self.data[key].append(value)
        args.seed = seed
        args.print()
        training.determine_gradient_accumulation(args)
        training.determine_train_batch_size(args)
        model = self.model_constructor(args)
        accs, _ = training.train(args, model, self.data_loaders)
        self.data['seed'].append(args.seed)
        self.data['train_acc'].append(accs['train'])
        self.data['dev_acc'].append(accs['dev'])
        self.data['test_acc'].append(accs['test'])
        df = pd.DataFrame(data=self.data, columns=self.columns)
        df.to_csv(grid_path(self.experiment_name), index=False)

    def evaluated(self, combination):
        if not os.path.exists(self.grid_path):
            return False
        df = pd.read_csv(self.grid_path)
        for key, value in combination.items():
            if isinstance(value, float):
                df = df[np.isclose(df[key], value)]
            else:
                df = df[df[key] == value]
        return len(df) > 0

    def get_or_load_data(self):
        # init the dict and columns
        data = {'id': []}
        columns = ['id']
        for key in self.search_keys:
            data[key] = []
            columns.append(key)
        data['seed'] = []
        data['train_acc'] = []
        data['dev_acc'] = []
        data['test_acc'] = []
        columns += ['seed', 'train_acc', 'dev_acc', 'test_acc']

        # load any old data
        if os.path.exists(self.grid_path):
            df = pd.read_csv(self.grid_path)
            data['id'] = list(df.id.values)
            for key in self.search_keys:
                data[key] = list(df[key].values)
            data['train_acc'] = list(df.train_acc.values)
            data['dev_acc'] = list(df.dev_acc.values)
            data['test_acc'] = list(df.test_acc.values)
            data['seed'] = list(df.seed.values)

        return data, columns

    @staticmethod
    def get_query(combination):
        query = ''
        for key, value in combination.items():
            if isinstance(value, str):
                value = "'%s'" % value
            else:
                value = str(value)
            query += ' & %s == %s' % (key, value)
        query = query[3:]
        return query

    @staticmethod
    def parse_dict(_dict):
        # wish I didn't need this hack for pandas
        # github issues reckons it should be solved in 24.0?
        keys = _dict.keys()
        values = []
        for value in _dict.values():
            if isinstance(value, np.bool_):
                value = bool(value)
            if isinstance(value, np.float64):
                value = float(value)
            if isinstance(value, np.int64):
                value = int(value)
            values.append(value)
        return dict(zip(keys, values))

    def winning_combinations(self):
        df = pd.read_csv(self.grid_path)
        best_acc = df.dev_acc.max()
        rows = df[df.dev_acc == best_acc]
        wanted_columns = list(self.search_keys) + ['id']
        column_selector = [c in wanted_columns for c in df.columns]
        if len(rows) > 1:  # have a tie break
            ids = rows.id.unique()
            ids_avgs = []
            for _id in ids:
                id_rows = df[df.id == _id]
                avg = id_rows.dev_acc.mean()
                ids_avgs.append((_id, avg))
            best_avg_acc = max(x[1] for x in ids_avgs)
            best_ids = [x[0] for x in ids_avgs if x[1] == best_avg_acc]
            combinations = []
            for _id in best_ids:
                rows = df[df.id == _id].loc[:, column_selector]
                combinations.append(rows.iloc[0].to_dict())
            best_acc = max(best_acc, best_avg_acc)
        else:
            rows = rows.loc[:, column_selector]
            combinations = [r[1].to_dict() for r in rows.iterrows()]
        combinations = [self.parse_dict(d) for d in combinations]
        return best_acc, combinations


def to_dict(df):
    data = {}
    for key, value in df.to_dict().items():
        data[key] = list(value.values())
    return data


def run(args, model_constructor, data_loaders, grid_space, n_experiments,
        train_fn=training.train, do_grid=True):
    # create the experiment folder if it doesn't already exist
    experiment_path = os.path.join(glovar.RESULTS_DIR, args.experiment_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    if do_grid:
        # conduct grid if best params not already found
        if not os.path.exists(params_path(args.experiment_name)):
            grid_search = GridSearch(
                experiment_name=args.experiment_name,
                model_constructor=model_constructor,
                data_loaders=data_loaders,
                args=args,
                search_space=grid_space)
            best_acc, best_params = grid_search()
        else:
            print('Loading best grid params...')
            with open(params_path(args.experiment_name), 'r') as f:
                best_params = json.loads(f.read())

        # merge best params
        for key, value in best_params.items():
            setattr(args, key, value)

    # determine gradient accumulation and effective batch size
    training.determine_gradient_accumulation(args)
    training.determine_train_batch_size(args)

    # run the experiments
    print('Running experiments...')

    # load or init new accs and preds data
    _accs_path = accs_path(args.experiment_name)
    if os.path.exists(_accs_path):
        accs = to_dict(pd.read_csv(_accs_path))
    else:
        accs = {
            'run_no': [],
            'seed': [],
            'train': [],
            'dev': [],
            'test': [],
            'test_adv': []}
    _preds_path = preds_path(args.experiment_name)
    if os.path.exists(_preds_path):
        preds = to_dict(pd.read_csv(_preds_path))
    else:
        preds = {
            'run_no': [],
            'dataset': [],
            'id': [],
            'prob0': [],
            'prob1': [],
            'pred': [],
            'correct': []}

    # conduct the experiments
    while len(accs['run_no']) < n_experiments:
        run_no = len(accs['run_no']) + 1

        # new random seed
        args.seed = random.choice(range(10000))
        while args.seed in accs['seed']:
            args.seed = random.choice(range(10000))
        
        # print info
        print('Experiment %s' % run_no)
        args.print()

        model = model_constructor(args)
        _accs, _preds = train_fn(args, model, data_loaders)

        # update accs
        accs['run_no'].append(run_no)
        accs['seed'].append(args.seed)
        accs['train'].append(_accs['train'])
        accs['dev'].append(_accs['dev'])
        accs['test'].append(_accs['test'])
        accs['test_adv'].append(_accs['test_adv'])

        # update preds
        for dataset in ['train', 'dev', 'test', 'test_adv']:
            for _pred in _preds[dataset]:
                preds['run_no'].append(run_no)
                preds['dataset'].append(dataset)
                preds['id'].append(_pred['id'])
                preds['prob0'].append(_pred['prob0'])
                preds['prob1'].append(_pred['prob1'])
                preds['pred'].append(_pred['pred'])
                preds['correct'].append(_pred['correct'])

        _accs = pd.DataFrame(data=accs, columns=accs.keys())
        _accs.to_csv(_accs_path, index=False)
        _preds = pd.DataFrame(data=preds, columns=preds.keys())
        _preds.to_csv(_preds_path, index=False)

    # report results
    print('Experiment results (%s):' % len(accs['train']))
    for dataset in ['train', 'dev', 'test', 'test_adv']:
        print('\t%s' % dataset)
        dataset_accs = accs[dataset]
        print('\t\tMean: %s' % np.mean(dataset_accs))
        print('\t\tMin:  %s' % min(dataset_accs))
        print('\t\tMax:  %s' % max(dataset_accs))
        print('\t\tStd:  %s' % np.std(dataset_accs))

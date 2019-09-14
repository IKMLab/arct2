"""Lists which experiments are complete, and which are not yet completed."""
import os

import pandas as pd

import glovar


if __name__ == '__main__':
    exp_names = sorted([x.replace('.py', '')
                        for x in os.listdir(glovar.EXPERIMENTS_DIR)
                        if x != '__pycache__'])
    yes = []
    no = []
    for name in exp_names:
        results_folder = os.path.join(glovar.RESULTS_DIR, name)
        if not os.path.exists(results_folder):
            no.append(f'{name}: no results folder yet.')
            continue
        accs_path = os.path.join(results_folder, 'accs.csv')
        if not os.path.exists(accs_path):
            no.append(f'{name}: no accs yet.')
            continue
        accs = pd.read_csv(accs_path)
        if len(accs) < 20:
            no.append(f'{name}: accs only has {len(accs)} results.')
            continue
        yes.append(f'{name}: '
                   f'{round(accs.test.min(), 3)}, '
                   f'{round(accs.test.mean(), 3)}, '
                   f'{round(accs.test.max(), 3)}')
    print('-' * 10)
    print('Complete:')
    print('-' * 10)
    for name in yes:
        print(name)
    print('-' * 10)
    print('Incomplete:')
    print('-' * 10)
    for name in no:
        print(name)

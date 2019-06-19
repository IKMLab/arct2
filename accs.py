"""View the accuracies for an experiment."""
import argparse
import pandas as pd
from util import experiments
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str)
    args = parser.parse_args()

    accs_path = experiments.accs_path(args.experiment_name)
    df = pd.read_csv(accs_path)

    print('Experiment results (%s):' % len(df))
    for dataset in ['train', 'dev', 'test']:
        print('\t%s' % dataset)
        print('\t\tMedian: %s' % df[dataset].median())
        print('\t\tMean:   %s' % df[dataset].mean())
        print('\t\tMin:    %s' % df[dataset].min())
        print('\t\tMax:    %s' % df[dataset].max())
        print('\t\tStd:    %s' % df[dataset].std())
    if 'test_adv' in df.columns:
        test_accs = df.test.values
        test_adv_accs = df.test_adv.values
        nc = np.ceil(test_accs * 444)
        nc_adv = np.ceil(test_adv_accs * 444)
        nc = nc[0:nc_adv.shape[0]] + nc_adv
        accs = nc / 888
        print('\tAdversarial Test')
        print('\t\tMedian: %s' % np.median(accs))
        print('\t\tMean:   %s' % np.mean(accs))
        print('\t\tMin:    %s' % np.min(accs))
        print('\t\tMax:    %s' % np.max(accs))
        print('\t\tStd:    %s' % np.std(accs))

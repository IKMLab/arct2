"""Script for running specific experiments.

Usage:

```
python run.py experiment_name
```
"""
import argparse
import importlib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment_name',
        type=str)
    args = parser.parse_args()

    experiment = importlib.import_module(
        'experiments.%s' % args.experiment_name)
    experiment.run()

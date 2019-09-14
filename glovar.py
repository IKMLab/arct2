"""Global variables."""
import os


PROJ_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CKPT_DIR = os.path.join(DATA_DIR, 'ckpts')
RESULTS_DIR = os.path.join(PROJ_DIR, 'results')
ARCT_DIR = os.path.join(DATA_DIR, 'arct')
GLOVE_PATH = os.path.join(DATA_DIR, 'glove', 'glove.840B.300d.txt')
EXPERIMENTS_DIR = os.path.join(PROJ_DIR, 'experiments')

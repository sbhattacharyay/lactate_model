### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import random
import datetime
import warnings
import itertools
import numpy as np
import pandas as pd
import pickle as cp
import seaborn as sns
import multiprocessing
from scipy import stats
from pathlib import Path
from ast import literal_eval
import matplotlib.pyplot as plt
from collections import Counter
from scipy.special import logit
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, recall_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler, minmax_scale
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# StatsModel methods
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

# TQDM for progress tracking
from tqdm import tqdm

# Define lactate data directory
lac_dir = '/home/sb2406/rds/hpc-work/lactate_data'

# Create directory to store model performance metrics
VERSION = 'v1-0'
model_dir = os.path.join(lac_dir,VERSION)
perf_dir = os.path.join(model_dir,'performance')
os.makedirs(perf_dir,exist_ok=True)

# Establish number of resamples for bootstrapping
NUM_RESAMP = 1000

# Load cross-validation information
compiled_partition_splits = pd.read_pickle('partition_splits.pickle')
study_id_outcome = compiled_partition_splits[['admissionid','LABEL']].drop_duplicates()

# If bootstrapping resamples don't exist, create them
if not os.path.exists(os.path.join(perf_dir,'bs_resamples.pkl')):
    
    # Make resamples for bootstrapping metrics
    bs_rs_GUPIs = []
    for i in range(NUM_RESAMP):
        np.random.seed(i)
        curr_GUPIs = np.unique(np.random.choice(study_id_outcome.admissionid,size=study_id_outcome.shape[0],replace=True))
        bs_rs_GUPIs.append(curr_GUPIs)
        
    # Create Data Frame to store bootstrapping resmaples 
    bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'ADMISSIONIDs':bs_rs_GUPIs})
    
    # Save bootstrapping resample dataframe
    bs_resamples.to_pickle(os.path.join(perf_dir,'bs_resamples.pkl'))
    
# Otherwise, load the pre-defined bootstrapping resamples
else:
    bs_resamples = pd.read_pickle(os.path.join(perf_dir,'bs_resamples.pkl'))

## Find all completed validation and testing set predictions
pred_files = []
for path in Path(os.path.join(model_dir)).rglob('*_predictions.csv'):
    pred_files.append(str(path.resolve()))
    
# Characterise list of discovered prediction files
pred_info_df = pd.DataFrame({'file':pred_files,
                             'VERSION':[re.search('_data/(.*)/tune', curr_file).group(1) for curr_file in pred_files],
                             'TUNE_IDX':[re.search('tune(.*)/uncalibrated_', curr_file).group(1) for curr_file in pred_files],
                             'SET':[re.search('uncalibrated_(.*)_predictions.csv', curr_file).group(1) for curr_file in pred_files]
                            }).sort_values(by=['TUNE_IDX','SET']).reset_index(drop=True)

# Iterate through 

### II.

    
# Load tuning grid of current model version
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))
set_grid = pd.DataFrame({'SET':['val','test'],'key':1})
tuning_grid['key'] = 1
bs_resamples['key'] = 1
rs_model_combos = pd.merge(bs_resamples,tuning_grid,how='outer',on='key').merge(set_grid,how='outer',on='key').drop(columns='key')

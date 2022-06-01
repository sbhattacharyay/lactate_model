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
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# TQDM for progress tracking
from tqdm import tqdm

# Custom analysis functions
from functions.analysis import collect_metrics

# Define directories in which performance metrics are saved
VERSION = 'v1-0'
performance_dir = '/home/sb2406/rds/hpc-work/lactate_data/'+VERSION+'/performance'

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Compile all performance metrics
# Search for all performance metric files in the directory
metric_files = []
for path in Path(os.path.join(performance_dir)).rglob('*.csv'):
    curr_string = str(path.resolve())
    if curr_string.endswith('orc.csv') | curr_string.endswith('threshold_metrics.csv'):
        metric_files.append(curr_string)

# Characterise list of discovered performance metric files
metric_info_df = pd.DataFrame({'file':metric_files,
                               'TUNE_IDX':[re.search('performance/tune(.*)/resample', curr_file).group(1) for curr_file in metric_files],
                               'RESAMPLE_IDX':[int(re.search('/resample(.*)/', curr_file).group(1)) for curr_file in metric_files],
                               'METRIC':[re.search('/resample(.*).csv', curr_file).group(1) for curr_file in metric_files]
                              }).sort_values(by=['TUNE_IDX','RESAMPLE_IDX','METRIC']).reset_index(drop=True)
metric_info_df['METRIC'] = metric_info_df['METRIC'].str.split('/').str[1]

# Iterate through unique metric types and compile APM_deep results into a single dataframe
for curr_metric in metric_info_df.METRIC.unique():
        
    # Filter files of current metric
    curr_metric_info_df = metric_info_df[metric_info_df.METRIC == curr_metric].reset_index(drop=True)
    
    # Partition current metric files among cores
    s = [curr_metric_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
    s[:(curr_metric_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(curr_metric_info_df.shape[0] - sum(s))]]    
    end_idx = np.cumsum(s)
    start_idx = np.insert(end_idx[:-1],0,0)

    # Collect current metric performance files in parallel
    curr_files_per_core = [(curr_metric_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Metric extraction: '+curr_metric) for idx in range(len(start_idx))]
    with multiprocessing.Pool(NUM_CORES) as pool:
        compiled_curr_metric_values = pd.concat(pool.starmap(collect_metrics, curr_files_per_core),ignore_index=True)
    
    # Calculate 95% confidence intervals
    if curr_metric.startswith('orc'):
        CI_overall = compiled_curr_metric_values.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False)['VALUE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'lo':lambda x: np.quantile(x,.025),'hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
        CI_overall.to_csv(os.path.join(performance_dir,'CI_ORC.csv'),index=False)

    elif curr_metric.startswith('threshold_metrics'):
        macro_compiled_threshold = compiled_curr_metric_values.groupby(['TUNE_IDX','WINDOW_IDX','RESAMPLE_IDX','METRIC'],as_index=False)['VALUE'].mean()
        macro_compiled_threshold['THRESHOLD'] = 'Average'
        compiled_curr_metric_values = pd.concat([compiled_curr_metric_values,macro_compiled_threshold],ignore_index=True)
        CI_threshold = compiled_curr_metric_values.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC'],as_index=False)['VALUE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'lo':lambda x: np.quantile(x,.025),'hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
        CI_threshold.to_csv(os.path.join(performance_dir,'CI_threshold_metrics.csv'),index=False)
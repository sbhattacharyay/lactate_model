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

# Custom methods
from functions.analysis import calc_thresh_calibration

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
test_splits = compiled_partition_splits[compiled_partition_splits.SET == 'test'].reset_index(drop=True)
study_id_outcome = test_splits[['admissionid','LABEL']].drop_duplicates()

# Load tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

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

## Calculate validation set performance
if not os.path.exists(os.path.join(perf_dir,'overall_validation_set_ORC.csv')):
    
    # Isolate validation predictions
    val_pred_info_df = pred_info_df[pred_info_df.SET == 'val'].reset_index(drop=True)

    # Load and compile validation predictions
    val_preds_df = pd.concat([pd.read_csv(f) for f in tqdm(val_pred_info_df.file,'Reading and compiling validation set predictions')],ignore_index=True)
    val_preds_df['WindowIdx'] = val_preds_df.groupby(['admissionid','TUNE_IDX']).cumcount(ascending=True)+1

    # Identify probability and logit columns
    prob_cols = [col for col in val_preds_df if col.startswith('Pr(SOFA_')]
    logit_cols = [col for col in val_preds_df if col.startswith('z_SOFA_')]

    # Calculate ordinal expected value
    prob_matrix = val_preds_df[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(3)), ndmin=2).T
    val_preds_df['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    val_preds_df['PredLabel'] = prob_matrix.idxmax(axis=1)

    # Get combination of tuning indices and window indices
    val_preds_combos = val_preds_df[['TUNE_IDX','WindowIdx']].drop_duplicates().reset_index(drop=True)

    # Iterate through validation prediction combinations and calculate ORC
    val_preds_combos['ORC'] = np.nan
    for curr_row in tqdm(range(val_preds_combos.shape[0]),'Calculating validation set performance for each tuning configuration'):

        curr_tune_idx = val_preds_combos.TUNE_IDX[curr_row]
        curr_window_idx = val_preds_combos.WindowIdx[curr_row]

        # Extract corresponding validation set predictions
        curr_val_preds = val_preds_df[(val_preds_df.TUNE_IDX == curr_tune_idx)&(val_preds_df.WindowIdx == curr_window_idx)].reset_index(drop=True)

        # Calculate current ORC
        orcs = []
        for ix, (a, b) in enumerate(itertools.combinations(np.sort(curr_val_preds.TrueLabel.unique()), 2)):
            curr_filt_val_preds = curr_val_preds[curr_val_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
            curr_filt_val_preds['ConditLabel'] = (curr_filt_val_preds.TrueLabel == b).astype(int)
            orcs.append(roc_auc_score(curr_filt_val_preds['ConditLabel'],curr_filt_val_preds['ExpectedValue']))
        curr_orc = np.mean(orcs)
        val_preds_combos.ORC[curr_row] = curr_orc

    # Save validation set ORCs
    val_preds_combos.to_csv(os.path.join(perf_dir,'validation_set_ORC.csv'),index=False)

    # Calculate overall ORC and last ORC by tuning index
    overall_val_ORC = val_preds_combos.groupby(['TUNE_IDX'],as_index=False)['ORC'].mean().sort_values('ORC',ascending=False).reset_index(drop=True).rename(columns={'ORC':'OverallORC'})
    last_val_ORC = val_preds_combos[val_preds_combos.WindowIdx == 24].sort_values('ORC',ascending=False).drop(columns='WindowIdx').reset_index(drop=True).rename(columns={'ORC':'LastORC'})
    val_ORC = overall_val_ORC.merge(last_val_ORC,how='left')
    val_ORC['AveORCScore']=(val_ORC['OverallORC']+val_ORC['LastORC'])/2
    val_ORC = val_ORC.sort_values(by=['AveORCScore'],ascending=False).reset_index(drop=True).merge(tuning_grid,how='left')
    val_ORC.to_csv(os.path.join(perf_dir,'overall_validation_set_ORC.csv'),index=False)

### II.
# Isolate testing predictions
test_pred_info_df = pred_info_df[pred_info_df.SET == 'test'].reset_index(drop=True)

# Load and compile testing predictions
test_preds_df = pd.concat([pd.read_csv(f) for f in tqdm(test_pred_info_df.file,'Reading and compiling testing set predictions')],ignore_index=True)
test_preds_df['WindowIdx'] = test_preds_df.groupby(['admissionid','TUNE_IDX']).cumcount(ascending=True)+1

# Identify probability and logit columns
prob_cols = [col for col in test_preds_df if col.startswith('Pr(SOFA_')]
logit_cols = [col for col in test_preds_df if col.startswith('z_SOFA_')]

# Calculate ordinal expected value
prob_matrix = test_preds_df[prob_cols]
prob_matrix.columns = list(range(prob_matrix.shape[1]))
index_vector = np.array(list(range(3)), ndmin=2).T
test_preds_df['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
test_preds_df['PredLabel'] = prob_matrix.idxmax(axis=1)

# Calculate cumulative probabilities at each threshold
thresh_labels = ['dSOFA>=0','dSOFA>=1']

for thresh in range(1,len(prob_cols)):
    cols_gt = prob_cols[thresh:]
    prob_gt = test_preds_df[cols_gt].sum(1).values
    gt = (test_preds_df['TrueLabel'] >= thresh).astype(int).values

    test_preds_df['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
    test_preds_df[thresh_labels[thresh-1]] = gt

## Argument-induced performance calculation functions
def main(array_task_id):
    
    # Get current resampling index and admission ID
    curr_rs = bs_resamples.RESAMPLE_IDX[array_task_id]
    curr_admissionids = bs_resamples.ADMISSIONIDs[array_task_id]

    # Get predictions of optimal validation index and current resample
    curr_is_preds = test_preds_df[(test_preds_df.TUNE_IDX == 1)&(test_preds_df.admissionid.isin(curr_admissionids))].reset_index(drop=True)

    # Create directory to save current combination outputs
    metric_dir = os.path.join(perf_dir,'tune'+str(1).zfill(4),'resample'+str(curr_rs).zfill(4))
    os.makedirs(metric_dir,exist_ok=True)
    
    # Define sequence of window indices for model assessment
    window_indices = list(range(1,25))

    # ORC
    orcs = []
    for curr_wi in tqdm(window_indices,'ORC'):
        filt_is_preds = curr_is_preds[curr_is_preds.WindowIdx == curr_wi].reset_index(drop=True)
        aucs = []
        for ix, (a, b) in enumerate(itertools.combinations(np.sort(filt_is_preds.TrueLabel.unique()), 2)):
            filt_prob_matrix = filt_is_preds[filt_is_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
            filt_prob_matrix['ConditLabel'] = (filt_prob_matrix.TrueLabel == b).astype(int)
            aucs.append(roc_auc_score(filt_prob_matrix['ConditLabel'],filt_prob_matrix['ExpectedValue']))
        orcs.append(pd.DataFrame({'TUNE_IDX':1,
                                  'RESAMPLE_IDX':curr_rs,
                                  'WINDOW_IDX':curr_wi,
                                  'METRIC':'ORC',
                                  'VALUE':np.mean(aucs)},index=[0]))
    orcs = pd.concat(orcs,ignore_index=True)
    
    ### Compile ORC into a single dataframe
    orcs.to_csv(os.path.join(metric_dir,'orc.csv'),index=False)
    
    ### Threshold-level AUC and ROC
    thresh_labels = ['dSOFA>=0','dSOFA>=1']
    thresh_aucs = []
    for curr_wi in tqdm(window_indices,'Threshold AUC'):
        filt_is_preds = curr_is_preds[curr_is_preds.WindowIdx == curr_wi].reset_index(drop=True)
        thresh_prob_labels = [col for col in filt_is_preds if col.startswith('Pr(dSOFA>=')]
        thresh_aucs.append(pd.DataFrame({'TUNE_IDX':1,
                                         'RESAMPLE_IDX':curr_rs,
                                         'WINDOW_IDX':curr_wi,
                                         'THRESHOLD':thresh_labels,
                                         'METRIC':'AUC',
                                         'VALUE':roc_auc_score(filt_is_preds[thresh_labels],filt_is_preds[thresh_prob_labels],average=None)}))
    thresh_aucs = pd.concat(thresh_aucs,ignore_index = True)
    
    ### Threshold-level calibration curves and associated metrics
    thresh_labels = ['dSOFA>=0','dSOFA>=1']
    calib_metrics = []
    for curr_wi in tqdm(window_indices,'Threshold Calibration'):
        filt_is_preds = curr_is_preds[curr_is_preds.WindowIdx == curr_wi].reset_index(drop=True)
        curr_thresh_metrics = calc_thresh_calibration(filt_is_preds)
        curr_thresh_metrics = curr_thresh_metrics.melt(id_vars=['THRESHOLD'], var_name='METRIC', value_name='VALUE')
        curr_thresh_metrics.insert(loc=0, column='TUNE_IDX', value=1)
        curr_thresh_metrics.insert(loc=1, column='RESAMPLE_IDX', value=curr_rs)
        curr_thresh_metrics.insert(loc=2, column='WINDOW_IDX', value=curr_wi)
        calib_metrics.append(curr_thresh_metrics)
    calib_metrics = pd.concat(calib_metrics,ignore_index = True).reset_index(drop=True)

    #### Compile and save threshold-level metrics
    thresh_level_metrics = pd.concat([thresh_aucs,calib_metrics],ignore_index=True)
    thresh_level_metrics.to_csv(os.path.join(metric_dir,'threshold_metrics.csv'),index=False)
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])
    main(array_task_id)
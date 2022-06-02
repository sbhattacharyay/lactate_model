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
from tqdm import tqdm
import seaborn as sns
import multiprocessing
from scipy import stats
from pathlib import Path
from ast import literal_eval
from scipy.special import logit
import matplotlib.pyplot as plt
from collections import Counter
from scipy.special import logit
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# StatsModel methods
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

# Function to calculate threshold-level calibration metrics
def calc_thresh_calibration(preds):
    
    prob_cols = [col for col in preds if col.startswith('Pr(SOFA_')]
    thresh_labels = ['dSOFA>=0','dSOFA>=1']
    calib_metrics = []
    
    for thresh in range(1,len(prob_cols)):
        cols_gt = prob_cols[thresh:]
        prob_gt = preds[cols_gt].sum(1).values
        gt = (preds['TrueLabel'] >= thresh).astype(int).values
        preds['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
        preds[thresh_labels[thresh-1]] = gt
        
    for thresh in thresh_labels:
        thresh_prob_name = 'Pr('+thresh+')'
        logit_gt = np.nan_to_num(logit(preds[thresh_prob_name]),neginf=-100,posinf=100)
        calib_glm = Logit(preds[thresh], add_constant(logit_gt))
        calib_glm_res = calib_glm.fit(disp=False)
        thresh_calib_linspace = np.linspace(preds[thresh_prob_name].min(),preds[thresh_prob_name].max(),200)
        TrueProb = lowess(endog = preds[thresh], exog = preds[thresh_prob_name], it = 0, xvals = thresh_calib_linspace)
        preds['TruePr('+thresh+')'] = preds[thresh_prob_name].apply(lambda x: TrueProb[(np.abs(x - thresh_calib_linspace)).argmin()])
        ICI = (preds['TruePr('+thresh+')'] - preds[thresh_prob_name]).abs().mean()
        Emax = (preds['TruePr('+thresh+')'] - preds[thresh_prob_name]).abs().max()
        calib_metrics.append(pd.DataFrame({'THRESHOLD':thresh,
                                           'CALIB_SLOPE':calib_glm_res.params[1],
                                           'ICI':ICI,
                                           'E_MAX':Emax},
                                         index=[0]))
        
    calib_metrics = pd.concat(calib_metrics,ignore_index = True).reset_index(drop=True)
    return calib_metrics

# Function to load and compile test performance metrics for models
def collect_metrics(metric_file_info,progress_bar = True, progress_bar_desc = ''):
    output_df = []
    if progress_bar:
        iterator = tqdm(metric_file_info.file,desc=progress_bar_desc)
    else:
        iterator = metric_file_info.file
    return pd.concat([pd.read_csv(f) for f in iterator],ignore_index=True)
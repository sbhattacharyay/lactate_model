### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import copy
import math
import random
import datetime
import warnings
import operator
import itertools
import functools
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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# TQDM for progress tracking
from tqdm import tqdm

# Import TimeSHAP methods
import timeshap.explainer as tsx
import timeshap.plot as tsp
from timeshap.wrappers import TorchModelWrapper
from timeshap.utils import get_avg_score_with_avg_event

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from models.dynamic_lactate import SOFA_model, timeshap_SOFA_model
from functions.model_building import format_shap, format_tokens, format_time_tokens, df_to_multihot_matrix

# TQDM for progress tracking
from tqdm import tqdm

# Define directories in which performance shaps are saved
VERSION = 'v1-0'

# Define model output directory based on version code
lac_dir = '/home/sb2406/rds/hpc-work/lactate_data'
model_dir = os.path.join(lac_dir,VERSION)

# Load the current version tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

thresh_labels = ['dSOFA>=0','dSOFA>=1']

# Define a directory for the storage of SHAP values
shap_dir = os.path.join(os.path.join(model_dir,'TimeSHAP'))

# Search for all SHAP files in the directory
shap_files = []
for path in Path(os.path.join(shap_dir)).rglob('*_Data.pkl'):
    shap_files.append(str(path.resolve()))

# Characterise list of discovered performance shap files
shap_info_df = pd.DataFrame({'file':shap_files,
                             'admissionid':[re.search('/AdmID_(.*)/Thresh_', curr_file).group(1) for curr_file in shap_files],
                             'ThresholdIdx':[int(re.search('/Thresh_(.*)_SHAP_', curr_file).group(1)) for curr_file in shap_files],
                             'Type':[re.search('_SHAP_(.*)_Data.pkl', curr_file).group(1) for curr_file in shap_files]
                            }).sort_values(by=['admissionid','ThresholdIdx','Type']).reset_index(drop=True)

# Iterate through and compile
event_info_df = shap_info_df[shap_info_df.Type == 'Event'].reset_index(drop=True)
event_TimeSHAP = []

for curr_event_row in tqdm(range(event_info_df.shape[0]),'Compiling event-specific TimeSHAP values'):
    
    curr_timeshap_values = pd.read_pickle(event_info_df.file[curr_event_row])
    curr_timeshap_values.insert(0,'admissionid',event_info_df.admissionid[curr_event_row])
    curr_timeshap_values.insert(1,'Threshold',thresh_labels[event_info_df.ThresholdIdx[curr_event_row]])
    
    curr_timeshap_values = curr_timeshap_values.drop(columns=['Random seed','NSamples']).rename(columns={'Feature':'HOURS_BEFORE_24_POST_ADM','Shapley Value':'SHAP'})
    event_TimeSHAP.append(curr_timeshap_values)
    
# Extract item id and item from token
event_TimeSHAP = pd.concat(event_TimeSHAP,ignore_index=True)
event_TimeSHAP.to_pickle(os.path.join(shap_dir,'Compiled_Timesteps_SHAP.pkl'))

# Iterate through and compile
feature_info_df = shap_info_df[shap_info_df.Type == 'Feature'].reset_index(drop=True)
feature_TimeSHAP = []

for curr_feature_row in tqdm(range(feature_info_df.shape[0]),'Compiling feature-specific TimeSHAP values'):
    
    curr_timeshap_values = pd.read_pickle(feature_info_df.file[curr_feature_row])
    curr_timeshap_values.insert(0,'admissionid',feature_info_df.admissionid[curr_feature_row])
    curr_timeshap_values.insert(1,'Threshold',thresh_labels[feature_info_df.ThresholdIdx[curr_feature_row]])
    
    curr_timeshap_values = curr_timeshap_values.drop(columns=['Random seed','NSamples']).rename(columns={'Feature':'TOKEN','Shapley Value':'SHAP'})
    feature_TimeSHAP.append(curr_timeshap_values)
    
# Extract item id and item from token
feature_TimeSHAP = pd.concat(feature_TimeSHAP,ignore_index=True)
feature_TimeSHAP.to_pickle(os.path.join(shap_dir,'Compiled_Features_SHAP.pkl'))
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

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample, shuffle
from sklearn.utils.class_weight import compute_class_weight

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

# Set version code
VERSION = 'v1-0'

# Define model output directory based on version code
lac_dir = '/home/sb2406/rds/hpc-work/lactate_data'
model_dir = os.path.join(lac_dir,VERSION)

# Load the current version tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

# Define a directory for the storage of SHAP values
shap_dir = os.path.join(os.path.join(model_dir,'TimeSHAP'))
os.makedirs(shap_dir,exist_ok=True)

# Define vector of SOFA thresholds
SOFA_thresholds = ['dSOFA>=0','dSOFA>=1']

# Load trained optimal model
sofa_model = SOFA_model.load_from_checkpoint(os.path.join(model_dir,'tune0001','epoch=06-val_loss=0.51.ckpt'))
sofa_model.eval()

# Load current vocabulary
curr_vocab = cp.load(open(os.path.join('/home/sb2406/rds/hpc-work/lactate_data','token_dictionary.pkl'),"rb"))
unknown_index = curr_vocab['<unk>']
    
# Extract current sets for current repeat and fold combination
training_set = pd.read_pickle(os.path.join(lac_dir,'training_set.pkl'))
validation_set = pd.read_pickle(os.path.join(lac_dir,'validation_set.pkl'))
testing_set = pd.read_pickle(os.path.join(lac_dir,'testing_set.pkl'))

# Format time tokens of index sets based on current tuning configuration
testing_set['SeqLength'] = testing_set.VocabIndex.apply(len)
testing_set['Unknowns'] = testing_set.VocabIndex.apply(lambda x: x.count(unknown_index))    

# Number of columns to add
cols_to_add = max(testing_set['Unknowns'].max(),1) - 1

# Define token labels from current vocab
token_labels = curr_vocab.get_itos() + [curr_vocab.get_itos()[unknown_index]+'_'+str(i+1).zfill(3) for i in range(cols_to_add)]
token_labels[unknown_index] = token_labels[unknown_index]+'_000'

# Convert testing set dataframes to multihot matrix
testing_multihot = df_to_multihot_matrix(testing_set, len(curr_vocab), unknown_index, cols_to_add)
testing_multihot_df = pd.DataFrame(testing_multihot.numpy(),columns=token_labels)
testing_multihot_df.insert(0,'admissionid',testing_set.admissionid.astype(str).str.zfill(5))
testing_multihot_df.insert(1,'WindowIdx',testing_set.WindowIdx)

# Calculate baseline ('average') values based on training set
flattened_training_set = training_set.groupby(['admissionid'],as_index=False).VocabIndex.aggregate(list)
flattened_training_set['IndexCounts'] = flattened_training_set.VocabIndex.apply(lambda x: [item for sublist in x for item in sublist]).apply(lambda x: dict(Counter(x)))
flattened_training_set['IndexCounts'] = flattened_training_set.apply(lambda x: {k: v / 24 for k, v in x.IndexCounts.items()}, axis=1)
IndexCounts = dict(functools.reduce(operator.add,map(Counter, flattened_training_set['IndexCounts'].to_list())))
IndexCounts = {k: v/flattened_training_set.shape[0] for k, v in IndexCounts.items() if (v/flattened_training_set.shape[0])>.5}
BaselineIndices = np.sort(list(IndexCounts.keys()))
AverageEvent = np.zeros([1,len(curr_vocab)+cols_to_add])
AverageEvent[0,BaselineIndices] = 1
AverageEvent = pd.DataFrame(AverageEvent,columns=token_labels).astype(int)

for curr_threshold_idx in tqdm(range(len(SOFA_thresholds)),'Calculating pruning statistics'):
    
    # Initialize custom TimeSHAP model
    ts_SOFA_model = timeshap_SOFA_model(sofa_model,curr_threshold_idx,unknown_index,cols_to_add)
    wrapped_sofa_model = TorchModelWrapper(ts_SOFA_model)
    f_hs = lambda x, y=None: wrapped_sofa_model.predict_last_hs(x, y)
    
    # Define pruning parameters
    pruning_dict = {'tol': [0.00625, 0.0125, 0.025, 0.05], 'path': os.path.join(shap_dir,'prun_all_thresh_idx_'+str(curr_threshold_idx)+'.csv')}
    prun_indexes = tsx.prune_all(f_hs, testing_multihot_df, 'admissionid', AverageEvent, pruning_dict, token_labels, 'WindowIdx')
    prun_indexes.to_pickle(os.path.join(shap_dir,'prun_idx_thresh_idx_'+str(curr_threshold_idx)+'.pkl'))
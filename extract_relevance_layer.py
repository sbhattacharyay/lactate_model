### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import copy
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

# Custom methods
from models.dynamic_lactate import SOFA_model

# Set version code
VERSION = 'v1-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/lactate_data/'+VERSION

# Load the current version tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

# Load trained optimal model
sofa_model = SOFA_model.load_from_checkpoint(os.path.join(model_dir,'tune0001','epoch=06-val_loss=0.51.ckpt'))
sofa_model.eval()

# Load current vocabulary
curr_vocab = cp.load(open(os.path.join('/home/sb2406/rds/hpc-work/lactate_data','token_dictionary.pkl'),"rb"))

# Extract relevance layer values
with torch.no_grad():
    relevance_layer = torch.exp(sofa_model.embedW.weight.detach().squeeze(1)).numpy()
    token_labels = curr_vocab.get_itos()        
    curr_relevance_df = pd.DataFrame({'TUNE_IDX':1,
                                      'TOKEN':token_labels,
                                      'RELEVANCE':relevance_layer})
    
# Extract item id and item from token
curr_relevance_df['ITEM'] = curr_relevance_df['TOKEN'].str.extract('(.*)_ITEMID')
curr_relevance_df['ITEM_ID'] = curr_relevance_df['TOKEN'].str.extract('_ITEMID(.*)_')

# Sort relevance layer by relevance value
curr_relevance_df = curr_relevance_df.sort_values(by='RELEVANCE',ascending=False).reset_index(drop=True)

# Save relevance layer values
curr_relevance_df.to_csv('/home/sb2406/rds/hpc-work/lactate_data/relevance_layer_weights.csv',index=False)
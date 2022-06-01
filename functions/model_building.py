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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")
from collections import Counter, OrderedDict
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# TQDM for progress tracking
from tqdm import tqdm

def collate_batch(batch):
    (label_list, idx_list, bin_offsets, gupi_offsets, gupis) = ([], [], [0], [0], [])
    for (seq_lists, curr_admissionid, curr_label) in batch:
        gupi_offsets.append(len(seq_lists))
        for curr_bin in seq_lists:
            label_list.append(curr_label)
            gupis.append(curr_admissionid)
            processed_bin_seq = torch.tensor(curr_bin,
                    dtype=torch.int64)
            idx_list.append(processed_bin_seq)
            bin_offsets.append(processed_bin_seq.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    gupi_offsets = torch.tensor(gupi_offsets[:-1]).cumsum(dim=0)
    bin_offsets = torch.tensor(bin_offsets[:-1]).cumsum(dim=0)
    idx_list = torch.cat(idx_list)
    return (label_list, idx_list, bin_offsets, gupi_offsets, gupis)

def categorizer(x,threshold=20):
    if is_integer_dtype(x) & (len(x.unique()) <= threshold):
        new_x = x.astype(str).str.zfill(3)
        new_x[new_x == 'nan'] = np.nan
        return new_x
    elif is_float_dtype(x) & (len(x.unique()) <= threshold):
        new_x = x.astype(str).str.replace('.','dec',regex=False)
        new_x[new_x.str.endswith('dec0')] = new_x[new_x.str.endswith('dec0')].str.replace('dec0','',regex=False)
        new_x = new_x.str.zfill(3)
        new_x[new_x == 'nan'] = np.nan
        return new_x
    else:
        return x
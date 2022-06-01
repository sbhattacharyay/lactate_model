# Fundamental libraries
import os
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
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# TQDM for progress tracking
from tqdm import tqdm

class DYN_ALL_PREDICTOR_SET(Dataset):
    def __init__(self, data):
        """
        Args:
            data (DataFrame): APM tokens pandas DataFrame
        """
        # Save arguments to class instance
        self.data = data
        
        # Get unique admissionid-OUTCOME information
        self.outcome_scores = self.data[['admissionid','OUTCOME']].drop_duplicates(ignore_index = True).sort_values(by=['admissionid']).reset_index(drop=True)
        
        # Initalize label encoder from Sci-Kit Learn and fit to all possible labels
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.outcome_scores.OUTCOME.values)
        
        
        self.y = self.outcome_scores.OUTCOME.values

        
        
        
        # Group by admissionid and concatenate complete sequence of 2-hour bin lists into an overall list per patient
        self.X = self.data.groupby(['admissionid'])['VocabIndex'].apply(list)
        
    # number of unique patients in the dataset
    def __len__(self):
        return len(self.y)

    # get a row at an index
    def __getitem__(self, idx):
        return self.X.iloc[idx], self.X.index[idx], self.y[idx]
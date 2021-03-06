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
    
def format_shap(shap_matrix,idx,token_labels,testing_set):
    shap_df = []
    for curr_pt_idx in range(shap_matrix.shape[0]):
        curr_pt_shap_matrix = shap_matrix[curr_pt_idx,:,:]
        curr_pt_shap_df = pd.DataFrame(curr_pt_shap_matrix,columns=token_labels)
        curr_pt_shap_df['GUPI'] = testing_set.GUPI.unique()[curr_pt_idx]
        curr_pt_shap_df['WindowIdx'] = list(range(1,curr_pt_shap_df.shape[0]+1))
        curr_pt_shap_df = curr_pt_shap_df.melt(id_vars = ['GUPI','WindowIdx'], var_name = 'Token', value_name = 'SHAP')
        curr_pt_shap_df['label'] = idx
        shap_df.append(curr_pt_shap_df)
    return pd.concat(shap_df,ignore_index=True)

def format_time_tokens(token_df,time_choice,train):
    
    # Extract indices pertaining to TFA and TOD tokens
    tfa_indices = token_df.VocabTimeFromAdmIndex.explode()
    tfa_indices = tfa_indices[tfa_indices != 0].astype(int).unique().tolist()

    tod_indices = token_df.VocabTimeOfDayIndex.explode()
    tod_indices = tod_indices[tod_indices != 0].astype(int).unique().tolist()
    
    if time_choice == 'None':
        
        token_df = token_df.drop(columns=['VocabTimeFromAdmIndex','VocabTimeOfDayIndex']).sort_values(by=['GUPI','WindowIdx'],ignore_index=True)
        mask_indices = tfa_indices + tod_indices
        
    elif time_choice == 'TFA_only':
        
        tfa_logicals = [idx_lst != [0] for idx_lst in token_df['VocabTimeFromAdmIndex']]
        token_df['VocabIndex'][tfa_logicals] = (token_df['VocabIndex'][tfa_logicals] + token_df['VocabTimeFromAdmIndex'][tfa_logicals])
        # If training set, clean 0 indices from VocabIndex w/ length more than 1
        if train:
            fix_logicals = [(len(idx_lst) > 1) & (0 in idx_lst) for idx_lst in token_df['VocabIndex']]
            token_df['VocabIndex'][fix_logicals] = token_df['VocabIndex'][fix_logicals].apply(lambda x: [i for i in x if i != 0])
        token_df = token_df.drop(columns=['VocabTimeFromAdmIndex','VocabTimeOfDayIndex']).sort_values(by=['GUPI','WindowIdx'],ignore_index=True)
        mask_indices = tod_indices

    elif time_choice == 'TOD_only':
        
        tod_logicals = [idx_lst != [0] for idx_lst in token_df['VocabTimeOfDayIndex']]
        token_df['VocabIndex'][tod_logicals] = (token_df['VocabIndex'][tod_logicals] + token_df['VocabTimeOfDayIndex'][tod_logicals])
        # If training set, clean 0 indices from VocabIndex w/ length more than 1
        if train:
            fix_logicals = [(len(idx_lst) > 1) & (0 in idx_lst) for idx_lst in token_df['VocabIndex']]
            token_df['VocabIndex'][fix_logicals] = token_df['VocabIndex'][fix_logicals].apply(lambda x: [i for i in x if i != 0])
        token_df = token_df.drop(columns=['VocabTimeFromAdmIndex','VocabTimeOfDayIndex']).sort_values(by=['GUPI','WindowIdx'],ignore_index=True)        
        mask_indices = tfa_indices
        
    elif time_choice == 'Both':
        
        tfa_logicals = [idx_lst != [0] for idx_lst in token_df['VocabTimeFromAdmIndex']]
        token_df['VocabIndex'][tfa_logicals] = (token_df['VocabIndex'][tfa_logicals] + token_df['VocabTimeFromAdmIndex'][tfa_logicals])
        tod_logicals = [idx_lst != [0] for idx_lst in token_df['VocabTimeOfDayIndex']]
        token_df['VocabIndex'][tod_logicals] = (token_df['VocabIndex'][tod_logicals] + token_df['VocabTimeOfDayIndex'][tod_logicals])
        # If training set, clean 0 indices from VocabIndex w/ length more than 1
        if train:
            fix_logicals = [(len(idx_lst) > 1) & (0 in idx_lst) for idx_lst in token_df['VocabIndex']]
            token_df['VocabIndex'][fix_logicals] = token_df['VocabIndex'][fix_logicals].apply(lambda x: [i for i in x if i != 0])
        token_df = token_df.drop(columns=['VocabTimeFromAdmIndex','VocabTimeOfDayIndex']).sort_values(by=['GUPI','WindowIdx'],ignore_index=True)        
        mask_indices = []
        
    return token_df, mask_indices

def T_scaling(logits, args):
    temperature = args.get('temperature', None)
    return torch.div(logits, temperature)

def vector_scaling(logits, args):
    curr_vector = args.get('vector', None)
    curr_biases = args.get('biases', None)
    return (torch.matmul(logits,torch.diag_embed(curr_vector.squeeze(1))) + curr_biases.squeeze(1))

def df_to_multihot_matrix(index_set, vocab_length, unknown_index, cols_to_add):
       
    # Initialize empty dataframe for multihot encoding
    multihot_matrix = np.zeros([index_set.shape[0],vocab_length+cols_to_add])
    
    # Encode testing set into multihot encoded matrix
    for i in tqdm(range(index_set.shape[0])):
        curr_indices = np.array(index_set.VocabIndex[i])
        if sum(curr_indices == unknown_index) > 1:
            zero_indices = np.where(curr_indices == unknown_index)[0]
            curr_indices[zero_indices[1:]] = [vocab_length + j for j in range(sum(curr_indices == unknown_index)-1)]
        multihot_matrix[i,curr_indices] = 1
    multihot_matrix = torch.tensor(multihot_matrix).float()
    
    return multihot_matrix


def format_tokens(token_df,window_lim,curr_adm_or_disch,window_dur):

    # Based on `curr_adm_or_disch` and window_lim, refine dataset
    if curr_adm_or_disch == 'adm':
        
        # Extract tokens up to window limit after ICU admission
        token_df = token_df[token_df.WindowIdx <= window_lim].sort_values(by=['GUPI','WindowIdx'],ignore_index=True)
        
        # Combine windows if `window_dur` does not equal 2
        if window_dur != 2:
            
            # Identify how many windows need to be combined per `window_dur` choice
            comb_span = int(window_dur/2)
            
            # Sequence of viable window indices
            window_indices_list = list(range(1,window_lim+1))

            # Split sequence of window indices
            windows_to_combine = [(window_indices_list[i:i+comb_span]) for i in range(0, len(window_indices_list), comb_span)]
            
            # Create empty list to store combined token indices
            combined_token_dfs = []
            
            # Iterate through window splits and combine token indices
            for group_idx, curr_group in enumerate(windows_to_combine):
                
                # Extract tokens in current group and combine by GUPI
                curr_token_group = token_df[token_df.WindowIdx.isin(curr_group)].reset_index(drop=True).groupby('GUPI')['VocabIndex'].apply(list).reset_index(name='VocabIndex')
                
                # Ensure tokens are unique within each combined window
                curr_token_group['VocabIndex'] = curr_token_group['VocabIndex'].apply(lambda x: list(np.unique([item for sublist in x for item in sublist])))
                
                # Extract timestamps for current token group
                curr_group_ts = token_df[token_df.WindowIdx.isin(curr_group)].drop(columns=['VocabIndex'])
                curr_group_ts = curr_group_ts.groupby(['GUPI','WindowTotal'],as_index=False).aggregate({'TimeStampStart':'min', 'TimeStampEnd':'max'}).reset_index(drop=True)
                
                # Change `WindowIdx` for new counting system
                curr_group_ts['WindowIdx'] = group_idx+1
                
                # Merge timestamps with token indices
                curr_token_group = pd.merge(curr_group_ts,curr_token_group,on='GUPI',how='left')
                
                # Append current token group to running list
                combined_token_dfs.append(curr_token_group)
                
            # Concatenate compiled token_dfs to form new token_df
            token_df = pd.concat(combined_token_dfs,ignore_index=True).sort_values(by=['GUPI','WindowIdx'],ignore_index=True)
            
            # Change `WindowTotal` to reflect combinations
            token_df.WindowTotal = np.ceil(token_df.WindowTotal/comb_span).astype(int)
            
    elif curr_adm_or_disch == 'disch':
        
        # Combine all tokens up to the window limit before ICU discharge
        comb_lim_tokens = token_df[token_df.WindowIdx >= window_lim].groupby('GUPI')['VocabIndex'].apply(list).reset_index(name='VocabIndex')
        comb_lim_tokens['VocabIndex'] = comb_lim_tokens['VocabIndex'].apply(lambda x: list(np.unique([item for sublist in x for item in sublist])))
        full_lim_training = token_df[token_df.WindowIdx == window_lim].drop(columns='VocabIndex').merge(comb_lim_tokens,on='GUPI',how='left')
        token_df = pd.concat([token_df[token_df.WindowIdx < window_lim],full_lim_training],ignore_index=True).sort_values(by=['GUPI','WindowIdx'],ascending=[True,False],ignore_index=True)
              
        # Combine windows if `window_dur` does not equal 2
        if window_dur != 2:
            
            # Identify how many windows need to be combined per `window_dur` choice
            comb_span = int(window_dur/2)
            
            # Sequence of viable window indices
            window_indices_list = list(range(1,window_lim+1))

            # Split sequence of window indices
            windows_to_combine = [(window_indices_list[i:i+comb_span]) for i in range(0, len(window_indices_list), comb_span)]
            
            # Create empty list to store combined token indices
            combined_token_dfs = []
            
            # Iterate through window splits and combine token indices
            for group_idx, curr_group in enumerate(windows_to_combine):
                
                # Extract tokens in current group and combine by GUPI
                curr_token_group = token_df[token_df.WindowIdx.isin(curr_group)].reset_index(drop=True).groupby('GUPI')['VocabIndex'].apply(list).reset_index(name='VocabIndex')
                
                # Ensure tokens are unique within each combined window
                curr_token_group['VocabIndex'] = curr_token_group['VocabIndex'].apply(lambda x: list(np.unique([item for sublist in x for item in sublist])))
                
                # Extract timestamps for current token group
                curr_group_ts = token_df[token_df.WindowIdx.isin(curr_group)].drop(columns=['VocabIndex'])
                curr_group_ts = curr_group_ts.groupby(['GUPI','WindowTotal'],as_index=False).aggregate({'TimeStampStart':'min', 'TimeStampEnd':'max'}).reset_index(drop=True)
                
                # Change `WindowIdx` for new counting system
                curr_group_ts['WindowIdx'] = group_idx+1
                
                # Merge timestamps with token indices
                curr_token_group = pd.merge(curr_group_ts,curr_token_group,on='GUPI',how='left')
                
                # Append current token group to running list
                combined_token_dfs.append(curr_token_group)
                
            # Concatenate compiled token_dfs to form new token_df
            token_df = pd.concat(combined_token_dfs,ignore_index=True).sort_values(by=['GUPI','WindowIdx'],ascending=[True,False],ignore_index=True)
            
            # Change `WindowTotal` to reflect combinations
            token_df.WindowTotal = np.ceil(token_df.WindowTotal/comb_span).astype(int)
        
    else:
        raise ValueError('curr_adm_or_disch must be "adm" or "disch"')
        
    return token_df
#### Master Script 1: Training full-lactate model ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II.

### I. Initialisation
## Import necessary packages
# Fundamental packages
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
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")
from collections import Counter, OrderedDict
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype

# SciKit-Learn methods
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab, Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# TimeSHAP methods
import timeshap.explainer as tsx
import timeshap.plot as tsp
from timeshap.wrappers import TorchModelWrapper
from timeshap.utils import get_avg_score_with_avg_event

# Custom methods
from classes.datasets import DYN_ALL_PREDICTOR_SET
from models.dynamic_lactate import SOFA_model
from functions.model_building import collate_batch, categorizer

## Define lactate data directory
lac_dir = '/home/sb2406/rds/hpc-work/lactate_data'

### II. Prepare study predictor sets
# Load study numeric predictors
numeric_2_predictors = pd.read_pickle(os.path.join(lac_dir,'df_numerical_2.pickle'))

# Load study categorical predictors
categorical_2_predictors = pd.read_pickle(os.path.join(lac_dir,'df_categorical_add_surgery.pickle'))

# Load baseline SOFA scores
baseline_SOFA = pd.read_pickle(os.path.join(lac_dir,'sofa_total_score_lactate_group.pickle')).rename(columns={'sofa_total_score':'BaselineSOFA'}).reset_index()

# Load endpoint SOFA scores
endpoint_SOFA = pd.read_pickle(os.path.join(lac_dir,'sofa_total_score_lactate_group_24_48.pickle')).rename(columns={'sofa_total_score':'EndpointSOFA'}).reset_index()

# Load admissions dataframe
admissions_df = pd.read_pickle(os.path.join(lac_dir,'admissions_df.pickle'))

# Add age group to categorical predictors
agegroups = admissions_df[['admissionid','agegroup']].rename(columns={'agegroup':'value'})
agegroups['item'] = 'AgeGroupAtAdmission'
agegroups['time_h_after_adm'] = np.nan
agegroups['itemid'] = -2
agegroups['type'] = 'age'
categorical_2_predictors = pd.concat([categorical_2_predictors,agegroups],ignore_index=True)

# Calculate delta SOFA
delta_SOFA = baseline_SOFA.merge(endpoint_SOFA,how='inner')
delta_SOFA['ChangeSOFA'] = delta_SOFA.EndpointSOFA - delta_SOFA.BaselineSOFA

# Filter out admission IDs within our population
numeric_2_predictors = numeric_2_predictors[numeric_2_predictors.admissionid.isin(delta_SOFA.admissionid)].reset_index(drop=True)
categorical_2_predictors = categorical_2_predictors[categorical_2_predictors.admissionid.isin(delta_SOFA.admissionid)].reset_index(drop=True)

# For items with missing item name, replace with surgery
categorical_2_predictors.item[categorical_2_predictors.item.isna()] = 'Surgery'
categorical_2_predictors.itemid[categorical_2_predictors.itemid.isna()] = -3

# Reassign predictor names
numeric_predictors = numeric_2_predictors.copy()
categorical_predictors = categorical_2_predictors.copy()

### III. Split dataset into training, validation, and testing
## Create change in SOFA label
# Categorize DeltaSOFA by directionality for stratified partitioning
delta_SOFA['LABEL'] = np.nan
delta_SOFA['LABEL'][delta_SOFA.ChangeSOFA < 0] = -1
delta_SOFA['LABEL'][delta_SOFA.ChangeSOFA == 0] = 0
delta_SOFA['LABEL'][delta_SOFA.ChangeSOFA > 0] = 1

# Load partition splits if they already exist
if os.path.exists('partition_splits.pickle'):
    compiled_partition_splits = pd.read_pickle('partition_splits.pickle')

# Otherwise, create the partition splits
else:
    
    # Set partition parameters
    TEST_CUT = 0.2
    VAL_CUT = 0.15

    # Initialize train-test stratified splitter with fixed random seed
    train_test_sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_CUT, random_state=12)

    # Extract testing and proto-training set
    for train_index, test_index in train_test_sss.split(delta_SOFA.drop(columns='LABEL'),delta_SOFA.LABEL.astype(int)):
        proto_training_set, testing_set = delta_SOFA.iloc[train_index].reset_index(drop=True), delta_SOFA.iloc[test_index].reset_index(drop=True)

    # Initialize train-val stratified splitter with fixed random seed
    train_val_sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_CUT/(1-TEST_CUT), random_state=2022)

    # Extract validation and training set
    for train_index, val_index in train_val_sss.split(proto_training_set.drop(columns='LABEL'),proto_training_set.LABEL.astype(int)):
        training_set, validation_set = proto_training_set.iloc[train_index].reset_index(drop=True), proto_training_set.iloc[val_index].reset_index(drop=True)

    # Save training, testing, and validation set information
    training_set['SET'] = 'train'
    validation_set['SET'] = 'val'
    testing_set['SET'] = 'test'
    compiled_partition_splits = pd.concat([training_set[['admissionid','ChangeSOFA','LABEL','SET']],validation_set[['admissionid','ChangeSOFA','LABEL','SET']],testing_set[['admissionid','ChangeSOFA','LABEL','SET']]],ignore_index=True)
    compiled_partition_splits.to_pickle('partition_splits.pickle')

### IV. Tokenize predictor set
# If tokenized testing set already exists, load all predictors
if (os.path.exists(os.path.join(lac_dir,'testing_set.pkl'))):
    curr_vocab = cp.load(open(os.path.join(lac_dir,'token_dictionary.pkl'),"rb"))
    train_indices = pd.read_pickle(os.path.join(lac_dir,'training_set.pkl'))
    val_indices = pd.read_pickle(os.path.join(lac_dir,'validation_set.pkl'))
    test_indices = pd.read_pickle(os.path.join(lac_dir,'testing_set.pkl'))
        
# Otherwise, tokenize
else:
    ## Tokenize numeric predictors
    # Define number of bins for discretization
    BINS = 20

    # Create empty lists to store tokenized values
    train_numeric_tokens = []
    val_numeric_tokens = []
    test_numeric_tokens = []

    # Isolate unique numeric predictors
    numeric_vars = numeric_predictors.item.unique()

    # Iterate through unique numeric predictors
    for curr_predictor in tqdm(numeric_vars,'Numeric predictors'):

        # Extract current partitions for current numeric predictor
        curr_train_values = numeric_predictors[(numeric_predictors.item == curr_predictor)&(numeric_predictors.admissionid.isin(compiled_partition_splits[compiled_partition_splits.SET=='train'].admissionid))].reset_index(drop=True)
        curr_val_values = numeric_predictors[(numeric_predictors.item == curr_predictor)&(numeric_predictors.admissionid.isin(compiled_partition_splits[compiled_partition_splits.SET=='val'].admissionid))].reset_index(drop=True)
        curr_test_values = numeric_predictors[(numeric_predictors.item == curr_predictor)&(numeric_predictors.admissionid.isin(compiled_partition_splits[compiled_partition_splits.SET=='test'].admissionid))].reset_index(drop=True)

        # Train k-bins discretizer on current numeric values
        curr_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        curr_train_values['BIN'] = curr_kbd.fit_transform(np.expand_dims(curr_train_values.value.values,1))+1
        curr_train_values[['unit','BIN']] = curr_train_values[['unit','BIN']].apply(categorizer,args=(100,))
        curr_train_values['TOKEN'] = (curr_train_values['item']+'_ITEMID'+curr_train_values['itemid'].astype(str).str.zfill(5)+'_BIN'+curr_train_values['BIN']).str.replace(r'\s+', '',regex=True)
        train_numeric_tokens.append(curr_train_values[['admissionid','time_h_after_adm','TOKEN']])

        # Discretize validation and testing sets
        if curr_val_values.shape[0] != 0:
            curr_val_values['BIN'] = curr_kbd.transform(np.expand_dims(curr_val_values.value.values,1))+1
            curr_val_values[['unit','BIN']] = curr_val_values[['unit','BIN']].apply(categorizer,args=(100,))
            curr_val_values['TOKEN'] = (curr_val_values['item']+'_ITEMID'+curr_val_values['itemid'].astype(str).str.zfill(5)+'_BIN'+curr_val_values['BIN']).str.replace(r'\s+', '',regex=True)
            val_numeric_tokens.append(curr_val_values[['admissionid','time_h_after_adm','TOKEN']])

        if curr_test_values.shape[0] != 0:
            curr_test_values['BIN'] = curr_kbd.transform(np.expand_dims(curr_test_values.value.values,1))+1
            curr_test_values[['unit','BIN']] = curr_test_values[['unit','BIN']].apply(categorizer,args=(100,))
            curr_test_values['TOKEN'] = (curr_test_values['item']+'_ITEMID'+curr_test_values['itemid'].astype(str).str.zfill(5)+'_BIN'+curr_test_values['BIN']).str.replace(r'\s+', '',regex=True)
            test_numeric_tokens.append(curr_test_values[['admissionid','time_h_after_adm','TOKEN']])

    # Concatenate numeric tokens per set
    train_numeric_tokens = pd.concat(train_numeric_tokens,ignore_index=True)
    val_numeric_tokens = pd.concat(val_numeric_tokens,ignore_index=True)
    test_numeric_tokens = pd.concat(test_numeric_tokens,ignore_index=True)

    ## Tokenize categorical predictors
    # Remove all formatting from categorical values
    categorical_predictors['itemid'] = categorical_predictors.itemid.astype(int)
    categorical_predictors['value'] = categorical_predictors.value.str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True).fillna('NAN')
    categorical_predictors['TOKEN'] = (categorical_predictors['item']+'_ITEMID'+categorical_predictors['itemid'].astype(str).str.zfill(5)+'_'+categorical_predictors['value']).str.replace(r'\s+', '',regex=True)

    # Split up categorical tokens by partition
    train_categorical_tokens = categorical_predictors[categorical_predictors.admissionid.isin(compiled_partition_splits[compiled_partition_splits.SET=='train'].admissionid)].reset_index(drop=True)[['admissionid','time_h_after_adm','TOKEN']]
    val_categorical_tokens = categorical_predictors[categorical_predictors.admissionid.isin(compiled_partition_splits[compiled_partition_splits.SET=='val'].admissionid)].reset_index(drop=True)[['admissionid','time_h_after_adm','TOKEN']]
    test_categorical_tokens = categorical_predictors[categorical_predictors.admissionid.isin(compiled_partition_splits[compiled_partition_splits.SET=='test'].admissionid)].reset_index(drop=True)[['admissionid','time_h_after_adm','TOKEN']]

    ## Train token dictionary and convert tokens to indices
    # Combine numeric and categorical tokens
    train_tokens = pd.concat([train_numeric_tokens,train_categorical_tokens],ignore_index=True)
    val_tokens = pd.concat([val_numeric_tokens,val_categorical_tokens],ignore_index=True)
    test_tokens = pd.concat([test_numeric_tokens,test_categorical_tokens],ignore_index=True)

    # Create an ordered dictionary to create a token vocabulary from admission
    training_token_list = (' '.join(train_tokens.TOKEN)).split(' ')
    if ('' in training_token_list):
        training_token_list = list(filter(lambda a: a != '', training_token_list))
    train_token_freqs = OrderedDict(Counter(training_token_list).most_common())

    # Build and save vocabulary (PyTorch Text) from admission
    curr_vocab = vocab(train_token_freqs, min_freq=1)
    null_token = ''
    unk_token = '<unk>'
    if null_token not in curr_vocab: curr_vocab.insert_token(null_token, 0)
    if unk_token not in curr_vocab: curr_vocab.insert_token(unk_token, len(curr_vocab))
    curr_vocab.set_default_index(curr_vocab[unk_token])
    cp.dump(curr_vocab, open(os.path.join(lac_dir,'token_dictionary.pkl'), "wb" ))

    # Convert training set tokens to indices
    train_tokens['VocabIndex'] = [curr_vocab[train_tokens.TOKEN[curr_row]] for curr_row in tqdm(range(train_tokens.shape[0]),desc='Converting training tokens to indices')]
    train_tokens = train_tokens.drop(columns='TOKEN')

    # Convert validation set tokens to indices
    val_tokens['VocabIndex'] = [curr_vocab[val_tokens.TOKEN[curr_row]] for curr_row in tqdm(range(val_tokens.shape[0]),desc='Converting validation tokens to indices')]
    val_tokens = val_tokens.drop(columns='TOKEN')

    # Convert testing set tokens to indices
    test_tokens['VocabIndex'] = [curr_vocab[test_tokens.TOKEN[curr_row]] for curr_row in tqdm(range(test_tokens.shape[0]),desc='Converting testing tokens to indices')]
    test_tokens = test_tokens.drop(columns='TOKEN')

    # Fix all negative hour tokens to baseline
    train_tokens.time_h_after_adm[train_tokens.time_h_after_adm < 0] = 0
    val_tokens.time_h_after_adm[val_tokens.time_h_after_adm < 0] = 0
    test_tokens.time_h_after_adm[test_tokens.time_h_after_adm < 0] = 0

    ## Bin tokens into 1-hour windows
    # Extract baseline tokens and group by patient
    baseline_train_tokens = train_tokens[train_tokens.time_h_after_adm.isna()].drop(columns='time_h_after_adm').reset_index(drop=True).drop_duplicates(subset=['admissionid','VocabIndex']).groupby(['admissionid'],as_index=False).VocabIndex.aggregate(list).rename(columns={'VocabIndex':'BaselineVocabIndex'})
    baseline_val_tokens = val_tokens[val_tokens.time_h_after_adm.isna()].drop(columns='time_h_after_adm').reset_index(drop=True).drop_duplicates(subset=['admissionid','VocabIndex']).groupby(['admissionid'],as_index=False).VocabIndex.aggregate(list).rename(columns={'VocabIndex':'BaselineVocabIndex'})
    baseline_test_tokens = test_tokens[test_tokens.time_h_after_adm.isna()].drop(columns='time_h_after_adm').reset_index(drop=True).drop_duplicates(subset=['admissionid','VocabIndex']).groupby(['admissionid'],as_index=False).VocabIndex.aggregate(list).rename(columns={'VocabIndex':'BaselineVocabIndex'})

    # Expand baseline predictors across all timestamps
    window_index_df = pd.DataFrame({'WindowIdx':np.linspace(1,24,24),'key':1})

    train_indices = pd.DataFrame({'admissionid':train_tokens.admissionid.unique(),'key':1}).merge(window_index_df,how='outer').merge(baseline_train_tokens,how='left').drop(columns='key').sort_values(by=['admissionid','WindowIdx'],ignore_index=True)
    val_indices = pd.DataFrame({'admissionid':val_tokens.admissionid.unique(),'key':1}).merge(window_index_df,how='outer').merge(baseline_val_tokens,how='left').drop(columns='key').sort_values(by=['admissionid','WindowIdx'],ignore_index=True)
    test_indices = pd.DataFrame({'admissionid':test_tokens.admissionid.unique(),'key':1}).merge(window_index_df,how='outer').merge(baseline_test_tokens,how='left').drop(columns='key').sort_values(by=['admissionid','WindowIdx'],ignore_index=True)

    # Bin dynamic event tokens
    dynamic_train_tokens = train_tokens[~train_tokens.time_h_after_adm.isna()].reset_index(drop=True)
    dynamic_val_tokens = val_tokens[~val_tokens.time_h_after_adm.isna()].reset_index(drop=True)
    dynamic_test_tokens = test_tokens[~test_tokens.time_h_after_adm.isna()].reset_index(drop=True)

    # Crate a time from admission discretizer
    tfa_kbd = KBinsDiscretizer(n_bins=24, encode='ordinal', strategy='uniform')

    # Cut 0 - 24 hours into 1 hour windows
    tfa_kbd.fit(np.expand_dims(np.linspace(0,24,10000),1))

    # Add timewindow information
    dynamic_train_tokens['WindowIdx'] = tfa_kbd.transform(np.expand_dims(dynamic_train_tokens.time_h_after_adm.values,1))+1
    dynamic_val_tokens['WindowIdx'] = tfa_kbd.transform(np.expand_dims(dynamic_val_tokens.time_h_after_adm.values,1))+1
    dynamic_test_tokens['WindowIdx'] = tfa_kbd.transform(np.expand_dims(dynamic_test_tokens.time_h_after_adm.values,1))+1

    # Ensure no duplicates in tokens and group by bins
    dynamic_train_tokens = dynamic_train_tokens.drop_duplicates(subset=['admissionid','WindowIdx','VocabIndex']).groupby(['admissionid','WindowIdx'],as_index=False).VocabIndex.aggregate(list)
    dynamic_val_tokens = dynamic_val_tokens.drop_duplicates(subset=['admissionid','WindowIdx','VocabIndex']).groupby(['admissionid','WindowIdx'],as_index=False).VocabIndex.aggregate(list)
    dynamic_test_tokens = dynamic_test_tokens.drop_duplicates(subset=['admissionid','WindowIdx','VocabIndex']).groupby(['admissionid','WindowIdx'],as_index=False).VocabIndex.aggregate(list)

    # Merge dynamic token information with the baseline
    train_indices = train_indices.merge(dynamic_train_tokens,how='left',on=['admissionid','WindowIdx'])
    val_indices = val_indices.merge(dynamic_val_tokens,how='left',on=['admissionid','WindowIdx'])
    test_indices = test_indices.merge(dynamic_test_tokens,how='left',on=['admissionid','WindowIdx'])

    # For non missing dynamic tokens, concatenate with baseline
    train_indices['BaselineVocabIndex'][~train_indices.VocabIndex.isna()] = train_indices['BaselineVocabIndex'][~train_indices.VocabIndex.isna()] + train_indices['VocabIndex'][~train_indices.VocabIndex.isna()] 
    val_indices['BaselineVocabIndex'][~val_indices.VocabIndex.isna()] = val_indices['BaselineVocabIndex'][~val_indices.VocabIndex.isna()] + val_indices['VocabIndex'][~val_indices.VocabIndex.isna()] 
    test_indices['BaselineVocabIndex'][~test_indices.VocabIndex.isna()] = test_indices['BaselineVocabIndex'][~test_indices.VocabIndex.isna()] + test_indices['VocabIndex'][~test_indices.VocabIndex.isna()] 

    # Reformat dataframes before saving
    train_indices = train_indices.drop(columns='VocabIndex').rename(columns={'BaselineVocabIndex':'VocabIndex'})
    val_indices = val_indices.drop(columns='VocabIndex').rename(columns={'BaselineVocabIndex':'VocabIndex'})
    test_indices = test_indices.drop(columns='VocabIndex').rename(columns={'BaselineVocabIndex':'VocabIndex'})

    # Save index files to drive
    train_indices.to_pickle(os.path.join(lac_dir,'training_set.pkl'))
    val_indices.to_pickle(os.path.join(lac_dir,'validation_set.pkl'))
    test_indices.to_pickle(os.path.join(lac_dir,'testing_set.pkl'))
    
### V. Train dynamic model based on selected tuning configuration
## Define modeling version and create model directory
VERSION = 'v1-0'
model_dir = os.path.join(lac_dir,VERSION)
os.makedirs(model_dir,exist_ok=True)

## If tuning grid does not yet exist, create it
if not os.path.exists(os.path.join(model_dir,'tuning_grid.csv')):

    # Create parameters for training differential token models
    tuning_parameters = {'LATENT_DIM':[32,64,128],
                         'HIDDEN_DIM':[32,64,128],
                         'RNN_TYPE':['LSTM','GRU'],
                         'EMBED_DROPOUT':[.2],
                         'RNN_LAYERS':[1],
                         'NUM_EPOCHS':[100],
                         'ES_PATIENCE':[10],
                         'IMBALANCE_CORRECTION':['weights'],
                         'OUTPUT_ACTIVATION':['softmax'],
                         'LEARNING_RATE':[0.001],
                         'BATCH_SIZE':[1,8,32]}
    
    # Convert parameter dictionary to dataframe
    tuning_grid = pd.DataFrame([row for row in itertools.product(*tuning_parameters.values())],columns=tuning_parameters.keys())
    tuning_grid['TUNE_IDX'] = list(range(1,tuning_grid.shape[0]+1))
    
    # Reorder tuning grid columns
    tuning_grid = tuning_grid[['TUNE_IDX','LATENT_DIM','HIDDEN_DIM','RNN_TYPE','EMBED_DROPOUT','RNN_LAYERS','NUM_EPOCHS','ES_PATIENCE','IMBALANCE_CORRECTION','OUTPUT_ACTIVATION','LEARNING_RATE','BATCH_SIZE']].reset_index(drop=True)
    
    # Save tuning grid to model directory
    tuning_grid.to_csv(os.path.join(model_dir,'tuning_grid.csv'),index=False)

else:
    # Load optimised tuning grid
    tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

## Argument-induced training functions
def main(array_task_id):
    
    # Collect parameters for training
    LATENT_DIM = tuning_grid.LATENT_DIM[array_task_id]
    HIDDEN_DIM = tuning_grid.HIDDEN_DIM[array_task_id]
    RNN_TYPE = tuning_grid.RNN_TYPE[array_task_id]
    EMBED_DROPOUT = tuning_grid.EMBED_DROPOUT[array_task_id]
    RNN_LAYERS = tuning_grid.RNN_LAYERS[array_task_id]
    LEARNING_RATE = tuning_grid.LEARNING_RATE[array_task_id]
    BATCH_SIZE = tuning_grid.BATCH_SIZE[array_task_id]
    ES_PATIENCE = tuning_grid.ES_PATIENCE[array_task_id]
    TUNE_IDX = tuning_grid.TUNE_IDX[array_task_id]
    NUM_EPOCHS = tuning_grid.NUM_EPOCHS[array_task_id]
    
    # Create current tuning directory
    tune_dir = os.path.join(model_dir,'tune'+str(TUNE_IDX).zfill(4))
    os.makedirs(tune_dir,exist_ok=True)

    # Load information
    curr_vocab = cp.load(open(os.path.join(lac_dir,'token_dictionary.pkl'),"rb"))
    train_indices = pd.read_pickle(os.path.join(lac_dir,'training_set.pkl'))
    val_indices = pd.read_pickle(os.path.join(lac_dir,'validation_set.pkl'))
    test_indices = pd.read_pickle(os.path.join(lac_dir,'testing_set.pkl'))
    
    # Merge with outcome information
    train_indices = train_indices.merge(delta_SOFA[['admissionid','LABEL']],how='left').rename(columns={'LABEL':'OUTCOME'})
    val_indices = val_indices.merge(delta_SOFA[['admissionid','LABEL']],how='left').rename(columns={'LABEL':'OUTCOME'})
    test_indices = test_indices.merge(delta_SOFA[['admissionid','LABEL']],how='left').rename(columns={'LABEL':'OUTCOME'})

    # Convert label to integer
    train_indices['OUTCOME'] = train_indices['OUTCOME'].astype(int)
    val_indices['OUTCOME'] = val_indices['OUTCOME'].astype(int)
    test_indices['OUTCOME'] = test_indices['OUTCOME'].astype(int)
    
    # Convert WindowIdx to integer
    train_indices['WindowIdx'] = train_indices['WindowIdx'].astype(int)
    val_indices['WindowIdx'] = val_indices['WindowIdx'].astype(int)
    test_indices['WindowIdx'] = test_indices['WindowIdx'].astype(int)

    # Load token dictionary
    curr_vocab = cp.load(open(os.path.join(lac_dir,'token_dictionary.pkl'),"rb"))

    # Create PyTorch Dataset objects
    train_Dataset = DYN_ALL_PREDICTOR_SET(train_indices)
    val_Dataset = DYN_ALL_PREDICTOR_SET(val_indices)
    test_Dataset = DYN_ALL_PREDICTOR_SET(test_indices)

    # Create PyTorch DataLoader objects
    curr_train_DL = DataLoader(train_Dataset,
                               batch_size=int(BATCH_SIZE),
                               shuffle=True,
                               collate_fn=collate_batch)

    curr_val_DL = DataLoader(val_Dataset,
                             batch_size=len(val_Dataset), 
                             shuffle=False,
                             collate_fn=collate_batch)

    curr_test_DL = DataLoader(test_Dataset,
                              batch_size=len(test_Dataset),
                              shuffle=False,
                              collate_fn=collate_batch)

    # Initialize current model class based on hyperparameter selections
    model = SOFA_model(len(curr_vocab),LATENT_DIM,EMBED_DROPOUT,RNN_TYPE,HIDDEN_DIM,RNN_LAYERS,LEARNING_RATE,True,train_Dataset.y,[0])

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=int(ES_PATIENCE),
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=tune_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    csv_logger = pl.loggers.CSVLogger(save_dir=model_dir,name='tune'+str(TUNE_IDX).zfill(4))

    trainer = pl.Trainer(gpus = 1,
                         accelerator='gpu',
                         logger = csv_logger,
                         max_epochs = int(NUM_EPOCHS),
                         enable_progress_bar = True,
                         enable_model_summary = True,
                         callbacks=[early_stop_callback,checkpoint_callback])

    trainer.fit(model=model,train_dataloaders=curr_train_DL,val_dataloaders=curr_val_DL)

    best_model = SOFA_model.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()

    ## Calculate and save uncalibrated validation set
    with torch.no_grad():
        for i, (curr_val_label_list, curr_val_idx_list, curr_val_bin_offsets, curr_val_gupi_offsets, curr_val_gupis) in enumerate(curr_val_DL):
            (val_yhat, out_val_gupi_offsets) = best_model(curr_val_idx_list, curr_val_bin_offsets, curr_val_gupi_offsets)
            curr_val_labels = torch.cat([curr_val_label_list],dim=0).cpu().numpy()
            curr_val_logits = torch.cat([val_yhat.detach()],dim=0).cpu().numpy()
            curr_val_probs = pd.DataFrame(F.softmax(torch.tensor(curr_val_logits)).cpu().numpy(),columns=['Pr(SOFA_dec)','Pr(SOFA_same)','Pr(SOFA_inc)'])
            curr_val_preds = pd.DataFrame(curr_val_logits,columns=['z_SOFA_dec','z_SOFA_same','z_SOFA_inc'])
            curr_val_preds = pd.concat([curr_val_preds,curr_val_probs], axis=1)
            curr_val_preds['TrueLabel'] = curr_val_labels
            curr_val_preds.insert(loc=0, column='admissionid', value=curr_val_gupis)        
            curr_val_preds['TUNE_IDX'] = TUNE_IDX
            curr_val_preds.to_csv(os.path.join(tune_dir,'uncalibrated_val_predictions.csv'),index=False)
            
    ## Calculate and save uncalibrated testing set
    with torch.no_grad():
        for i, (curr_test_label_list, curr_test_idx_list, curr_test_bin_offsets, curr_test_gupi_offsets, curr_test_gupis) in enumerate(curr_test_DL):
            (test_yhat, out_test_gupi_offsets) = best_model(curr_test_idx_list, curr_test_bin_offsets, curr_test_gupi_offsets)
            curr_test_labels = torch.cat([curr_test_label_list],dim=0).cpu().numpy()
            curr_test_logits = torch.cat([test_yhat.detach()],dim=0).cpu().numpy()
            curr_test_probs = pd.DataFrame(F.softmax(torch.tensor(curr_test_logits)).cpu().numpy(),columns=['Pr(SOFA_dec)','Pr(SOFA_same)','Pr(SOFA_inc)'])
            curr_test_preds = pd.DataFrame(curr_test_logits,columns=['z_SOFA_dec','z_SOFA_same','z_SOFA_inc'])
            curr_test_preds = pd.concat([curr_test_preds,curr_test_probs], axis=1)
            curr_test_preds['TrueLabel'] = curr_test_labels
            curr_test_preds.insert(loc=0, column='admissionid', value=curr_test_gupis)        
            curr_test_preds['TUNE_IDX'] = TUNE_IDX
            curr_test_preds.to_csv(os.path.join(tune_dir,'uncalibrated_test_predictions.csv'),index=False)
            
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import os
import gc
import re
import math
import time
import random
import shutil
import pickle
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import Levenshtein
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau


import warnings 
warnings.filterwarnings('ignore')

from preprocessing import Tokenizer, TestDataset
from model import TopKDecoder, Encoder, DecoderWithAttention
from utils import get_test_file_path, get_train_file_path, init_logger, seed_torch, get_score, get_transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load Tokenizer
tokenizer = torch.load('../input/inchi-preprocess-2/tokenizer2.pth')
print(f"tokenizer.stoi: {tokenizer.stoi}")

# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    max_len=275
    print_freq=1000
    num_workers=4
    model_name='resnet34'
    size=224
    scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs=1 # not to exceed 9h
    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    T_max=4 # CosineAnnealingLR
    #T_0=4 # CosineAnnealingWarmRestarts
    encoder_lr=1e-4
    decoder_lr=4e-4
    min_lr=1e-6
    batch_size=256
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5
    seed=42
    n_fold=5
    trn_fold=[0] # [0, 1, 2, 3, 4]
    train=True

LOGGER = init_logger()
seed_torch(seed=CFG.seed)

# ====================================================
# Inference
# ====================================================
def inference(test_loader, encoder, decoder, tokenizer, device):
    encoder.eval()
    decoder.eval()
    text_preds = []
    
    # k = 2
    topk_decoder = TopKDecoder(decoder, 2, CFG.decoder_dim, CFG.max_len, tokenizer)
    
    tk0 = tqdm(test_loader, total=len(test_loader))
    for images in tk0:
        images = images.to(device)
        predictions = []
        with torch.no_grad():
            encoder_out = encoder(images)
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
            h, c = decoder.init_hidden_state(encoder_out)
            hidden = (h.unsqueeze(0), c.unsqueeze(0))
            
            decoder_outputs, decoder_hidden, other = topk_decoder(None, hidden, encoder_out)
            
            for b in range(batch_size):
                length = other['topk_length'][b][0]
                tgt_id_seq = [other['topk_sequence'][di][b, 0, 0].item() for di in range(length)]
                predictions.append(tgt_id_seq)
            assert len(predictions) == batch_size
            
        predictions = tokenizer.predict_captions(predictions)
        predictions = ['InChI=1S/' + p.replace('<sos>', '') for p in predictions]
        # print(predictions[0])
        text_preds.append(predictions)
    text_preds = np.concatenate(text_preds)
    return text_preds

test = pd.read_csv('../input/bms-molecular-translation/sample_submission.csv')
test['file_path'] = test['image_id'].apply(get_test_file_path)
print(f'test.shape: {test.shape}')

states = torch.load(f'../input/inchiresnetlstmwithattentionstarter10epochs/resnet34-fold0-10epochs.pth', map_location=torch.device('cpu'))

encoder = Encoder(CFG.model_name, pretrained=False)
encoder.load_state_dict(states['encoder'])
encoder.to(device)

decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                               embed_dim=CFG.embed_dim,
                               decoder_dim=CFG.decoder_dim,
                               vocab_size=len(tokenizer),
                               dropout=CFG.dropout,
                               device=device)
decoder.load_state_dict(states['decoder'])
decoder.to(device)

del states; gc.collect()

test_dataset = TestDataset(test, transform=get_transforms(CFG.size, data='valid'))
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=CFG.num_workers)
predictions = inference(test_loader, encoder, decoder, tokenizer, device)

del test_loader, encoder, decoder, tokenizer; gc.collect()
# submission
test['InChI'] = [f"{text}" for text in predictions]
test[['image_id', 'InChI']].to_csv('submission.csv', index=False)
test[['image_id', 'InChI']].head()
import os
import numpy as np
import random
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pathlib import Path

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import Levenshtein


# ====================================================
# Utils
# ====================================================
def get_test_file_path(image_id):
    return "../input/bms-molecular-translation/test/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id 
    )

def get_train_file_path(image_id):
    return "../input/bms-molecular-translation/train/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id 
    )

def get_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score


def init_logger(log_file='inference.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False # True

# ====================================================
# Transforms
# ====================================================
def get_transforms(size, data):
    
    if data == 'train':
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif data == 'valid':
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
# ====================================================
# Nomalization
# ====================================================
def _normalize_inchi(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        return inchi if (mol is None) else Chem.MolToInchi(mol)
    except: return inchi

def nomalization_run(orig_path):
    # Input & Output
    norm_path = orig_path.with_name(orig_path.stem+'_norm.csv')
    
    # Do the job
    N = norm_path.read_text().count('\n') if norm_path.exists() else 0
    print(N, 'number of predictions already normalized')

    r = open(str(orig_path), 'r')
    w = open(str(norm_path), 'a', buffering=1)

    for _ in range(N):
        r.readline()
    line = r.readline()  # this line is the header or is where it segfaulted last time
    w.write(line)

    pbar = tqdm()
    while True:
        line = r.readline()
        if not line:
            break  # done
        image_id = line.split(',')[0]
        inchi = ','.join(line[:-1].split(',')[1:]).replace('"','')
        inchi_norm = _normalize_inchi(inchi)
        w.write(f'{image_id},"{inchi_norm}"\n')
        pbar.update(1)
    r.close()
    w.close()
    print('Done Nomalization')
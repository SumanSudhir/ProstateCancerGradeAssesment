import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

import numpy as np
import pandas as pd
import random

import PIL
from PIL import Image
import skimage.io
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F

from dataset.dataset import *
from net.modules import *


SEED = 2021
size = 256
split = 0
nfolds = 4
DEBUG = False

files_path = '../../CLAM/pandaPatches10x/patches'
train_csv = '../../data/train.csv'
weight = '../../data/radbound_weight_tkarolinska_0.csv'

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

seed_everything(SEED)

train_on_gpu = torch.cuda.is_available()
# train_on_gpu = False

if not train_on_gpu:
    print("CUDA is not available. Training on CPU...")
else:
    print("CUDA is available. Training on GPU...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


df = pd.read_csv(train_csv)
#
# files = sorted(set([p[:-3] for p in os.listdir(files_path) if p.endswith('.h5')]))
# df = df.loc[files]
# df = df.reset_index()

# df = df[df['isup_grade'] != 0]
df = df[df['data_provider'] != 'karolinska']

# splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
# splits = list(splits.split(df, df.isup_grade))
# folds_splits = np.zeros(len(df)).astype(np.int)
#
# weighted_df = pd.read_csv(weight)
#
# for i in range(nfolds): folds_splits[splits[i][1]] = i
# df["split"] = folds_splits

print("Previous Length", len(df))
if DEBUG:
    df = df[:100]

print("Usable Length", len(df))

"""Mean and Std deviation"""
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

"""Dataset"""

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

v_dataset = Whole_Slide_ROI(df, weighted_df, files_path, valid_transform)
validloader = DataLoader(v_dataset, batch_size=6, shuffle=False, num_workers=4)


"""Training"""
model = EfficientAvgModel()
model.to(device)
model.load_state_dict(torch.load('../OUTPUT/stage2/split_0/efficient_b0_24_0.8203206327782633.pth', map_location=torch.device(device)))
model.eval()

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()

validation_loss = []
validation_loss = []
valid_pred = []
valid_label = []

model.eval()
avg_valid_loss = 0.0
with torch.no_grad():
    for img,label,_,_ in tqdm(validloader):
        if train_on_gpu:
            img, label = img.to(device), label.to(device)

        logits = model.forward(img)

        val_loss = criterion(logits.float(), label.float())
        avg_valid_loss += val_loss.item()

        pred = logits.sigmoid().sum(1).detach().round().cpu()
        valid_pred.append(pred)
        valid_label.append(label.sum(1).cpu())

valid_pred = torch.cat(valid_pred).cpu().numpy()
valid_label = torch.cat(valid_label).cpu().numpy()

valid_cm = np.array(confusion_matrix(valid_label, valid_pred))

avg_valid_loss /= len(validloader)
valid_acc = accuracy_score(valid_label, valid_pred)
score = cohen_kappa_score(valid_label, valid_pred, weights='quadratic')

print("Validation Accuracy:", valid_acc)
print("Kappa Score", score)

print(valid_cm)

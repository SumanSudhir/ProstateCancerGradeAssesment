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

from torch.utils.tensorboard import SummaryWriter

SEED = 2021
size = 96
split = 0
nfolds = 4
N = 64
DEBUG = False

# files_path = '../../CLAM/pandaPatches10x/patches'
# train_csv = '../../data/train.csv'

files_path = '../../CLAM/TCGA/patches'
# train_csv = '../../data/prostate_tcga.csv'
train_csv = '../../data/tcga_new_isup.csv'



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


## TCGA
df = pd.read_csv(train_csv).set_index('bcr_patient_barcode')
files = sorted(set([p for p in os.listdir(files_path) if p.endswith('.h5')]))
df['image_id'] = 'New'
for i in files:
    x = i[:12]
    df.at[x,'image_id'] = i[:-3]

# df = df.reset_index().set_index('image_id')

## General
# files = sorted(set([p[:-3] for p in os.listdir(files_path) if p.endswith('.h5')]))
files = sorted(set([p[:12] for p in os.listdir(files_path) if p.endswith('.h5')]))
df = df.loc[files]
df = df.reset_index()

# print(df)
#
df = df[df['isup_grade'] != 0]
# df = df[df['data_provider'] != 'karolinska']
df = df[df['tissue_source_site'] == 'HC']

splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
splits = list(splits.split(df, df.isup_grade))
folds_splits = np.zeros(len(df)).astype(np.int)

for i in range(nfolds): folds_splits[splits[i][1]] = i
df["split"] = folds_splits

print("Previous Length", len(df))
if DEBUG:
    df = df[:20]

print("Usable Length", len(df))

"""Mean and Std deviation"""
# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

"""Dataset"""

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

# df_train = df[df["split"] != split]
# df_valid = df[df["split"] == split]

v_dataset = Whole_Slide_Bag(df, files_path, valid_transform, num_patches=N)
validloader = DataLoader(v_dataset, batch_size=6, shuffle=False, num_workers=4)




#
model = EfficientModel(c_out=5, tile_size=size, n_tiles=N)
model.to(device)
model.load_state_dict(torch.load('../OUTPUT/karolinska/stage1/split_1/efficient_b0_18_0.8093723432575634.pth', map_location=torch.device(device)))
model.eval()


img_ids = []
tile_ids = []
att_w = []
Xc = []
Yc = []

valid_pred = []
valid_label = []

for idx,(img,label,_,coords,image_id)  in enumerate(tqdm((validloader))):
    if train_on_gpu:
        img, label = img.to(device), label.to(device)

    model.eval()
    with torch.no_grad():

        logits1, weight, instance_loss1 = model(img)
        logits2, weight, instance_loss2 = model(img.flip(-1))

        logits = 0.5*logits1 + 0.5*logits2
        pred = logits.sigmoid().sum(1).detach().round()

        # weight = weight.squeeze().cpu().numpy()
        # for i in range(len(weight)):
        #     img_ids.append(image_id[0])
        #     tile_ids.append(i)
        #     att_w.append(weight[i])
        #     Xc.append(coords[i].tolist()[0][0])
        #     Yc.append(coords[i].tolist()[0][1])

    valid_pred.append(pred.cpu().numpy())
    valid_label.append(label.sum(1).cpu().numpy())

valid_pred1 = np.concatenate(valid_pred, axis=None)
valid_label1 = np.concatenate(valid_label, axis=None)

# img_ids = np.concatenate(img_ids, axis=None)
# tile_ids = np.concatenate(tile_ids, axis=None)
# att_w = np.concatenate(att_w, axis=None)
# Xc = np.concatenate(Xc, axis=None)
# Yc = np.concatenate(Yc, axis=None)


# valid_pred1, valid_label1 = [],[]
# for i in range(len(valid_pred)):
#     if(valid_pred[i] != 0):
#         valid_pred1.append(valid_pred[i]-1)
#         valid_label1.append(valid_label[i]-1)


# valid_pred1 = [i-1 for i in valid_pred if i != 0]
# valid_label1 = [i-1 for i in valid_label if i != 0]

valid_cm = np.array(confusion_matrix(valid_label1, valid_pred1))

valid_acc = accuracy_score(valid_label1, valid_pred1)
score = cohen_kappa_score(valid_label1, valid_pred1, weights='quadratic')

# sub_df = pd.DataFrame({'image_id': img_ids, 'tile_id': tile_ids, 'weight': att_w, 'X': Xc, 'Y': Yc})
# sub_df.to_csv('../../data/tcga_weight_tpanda_0.csv', index=False)

print("Validation Accuracy:", valid_acc)
print("Kappa Score", score)
print(valid_cm)

#####################################################################

# model_1 = EfficientModel(c_out=4, tile_size=size, n_tiles=N)
# model_1.load_state_dict(torch.load('../OUTPUT/tcga/stage1/split_0/efficient_b0_20_0.6779463243873979.pth', map_location=torch.device(device)))
# # model_1.load_state_dict(torch.load('../OUTPUT/radbound/stage1/split_0/efficient_b0_17_0.6364211353668056.pth', map_location=torch.device(device)))
# # model_1.load_state_dict(torch.load('../OUTPUT/stage1/split_0/efficient_b0_17_0.6419193685015757.pth', map_location=torch.device(device)))
# model_1.load_state_dict(torch.load('../OUTPUT/radboud/stage1/split_3/efficient_b0_12_0.7296633941093968.pth', map_location=torch.device(device)))
# model_1.to(device)
# model_1.eval()
#
#
# model = AdaptNet(tile_size=size)
# model.load_state_dict(torch.load('../OUTPUT/radboud/adapt/split_3/adapt_128_50_0.000691256259978261.pth', map_location=torch.device(device)))
# # model = nn.DataParallel(model, device_ids=[2,3])
# model.to(device)
#
#
# img_ids = []
# tile_ids = []
# att_w = []
# Xc = []
# Yc = []
#
# valid_pred = []
# valid_label = []
#
# for idx,(img,label,_,coords,image_id)  in enumerate(tqdm((validloader))):
#     if train_on_gpu:
#         img, label = img.to(device), label.to(device)
#
#     model_1.eval()
#     model.eval()
#     with torch.no_grad():
#         # logits, weight, instance_loss = model_1(img)
#         logits1, _, instance_loss1 = model_1(model(img.view(-1,3,size,size).to(device)))
#         logits2, _, instance_loss2 = model_1(model(img.view(-1,3,size,size).to(device)).flip(-1))
#
#         logits = 0.5*logits1 + 0.5*logits2
#         instance_loss = 0.5*instance_loss1 + 0.5*instance_loss2
#         pred = logits.sigmoid().sum(1).detach().round()
#
#         # weight = weight.squeeze().cpu().numpy()
#         # for i in range(len(weight)):
#         #     img_ids.append(image_id[0])
#         #     tile_ids.append(i)
#         #     att_w.append(weight[i])
#         #     Xc.append(coords[i].tolist()[0][0])
#         #     Yc.append(coords[i].tolist()[0][1])
#
#     valid_pred.append(pred.cpu().numpy())
#     valid_label.append(label.sum(1).cpu().numpy())
#
# valid_pred = np.concatenate(valid_pred, axis=None)
# valid_label = np.concatenate(valid_label, axis=None)
#
# # img_ids = np.concatenate(img_ids, axis=None)
# # tile_ids = np.concatenate(tile_ids, axis=None)
# # att_w = np.concatenate(att_w, axis=None)
# # Xc = np.concatenate(Xc, axis=None)
# # Yc = np.concatenate(Yc, axis=None)
#
# valid_cm = np.array(confusion_matrix(valid_label, valid_pred))
#
# valid_acc = accuracy_score(valid_label, valid_pred)
# score = cohen_kappa_score(valid_label, valid_pred, weights='quadratic')
#
# # sub_df = pd.DataFrame({'image_id': img_ids, 'tile_id': tile_ids, 'weight': att_w, 'X': Xc, 'Y': Yc})
# # sub_df.to_csv('../../data/karolinska_weight_tcga_0.csv', index=False)
#
# print("Validation Accuracy:", valid_acc)
# print("Kappa Score", score)
#
# print(valid_cm)

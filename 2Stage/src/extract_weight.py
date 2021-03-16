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
size = 128
N = 64
split = 0
nfolds = 4
epochs = 20
DEBUG = False


files_path = '../../CLAM/pandaPatches10x/patches'
train_csv = '../../data/train.csv'

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


df = pd.read_csv(train_csv).set_index('image_id')

files = sorted(set([p[:-3] for p in os.listdir(files_path) if p.endswith('.h5')]))
df = df.loc[files]
df = df.reset_index()

# df = df[df['isup_grade'] != 0]
df = df[df['data_provider'] != 'karolinska']

# splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
# splits = list(splits.split(df, df.isup_grade))
# folds_splits = np.zeros(len(df)).astype(np.int)
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
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

# df_train = df[df["split"] != split]
# df_valid = df[df["split"] == split]

v_dataset = Whole_Slide_Bag(df, files_path, valid_transform, num_patches=N)
validloader = DataLoader(v_dataset, batch_size=10, shuffle=False, num_workers=4)

model = EfficientModel(c_out=5, tile_size=size, n_tiles=N)
model.to(device)
# model_1.load_state_dict(torch.load('../OUTPUT/tcga/stage1/split_0/efficient_b0_20_0.6779463243873979.pth', map_location=torch.device(device)))
# model_1.load_state_dict(torch.load('../OUTPUT/radbound/stage1/split_0/efficient_b0_17_0.6364211353668056.pth', map_location=torch.device(device)))
# model_1.load_state_dict(torch.load('../OUTPUT/stage1/split_0/efficient_b0_17_0.6419193685015757.pth', map_location=torch.device(device)))
# model_1.load_state_dict(torch.load('../OUTPUT/radboud/stage1/split_3/efficient_b0_12_0.7296633941093968.pth', map_location=torch.device(device)))
# model_1.to(device)
# model_1.eval()


path1 = '../OUTPUT/karolinska/stage1n//split_0/efficient_b0_64_26_0.8465.pth'
path2 = '../OUTPUT/karolinska/stage1n//split_1/efficient_b0_64_19_0.8268.pth'

states = [torch.load(path1), torch.load(path2)]
model.load_state_dict(states[0])
model.eval()

# model = AdaptNet(tile_size=size)
# model.load_state_dict(torch.load('../OUTPUT/radboud/adapt/split_3/adapt_128_50_0.000691256259978261.pth', map_location=torch.device(device)))
# # model = nn.DataParallel(model, device_ids=[2,3])
# model.to(device)


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

    with torch.no_grad():
        y_preds, weight, _ = model(img)

    weight = weight.squeeze().cpu().numpy()
    for i,ids in enumerate(image_id):
        for j in range(N):
            img_ids.append(ids)
            tile_ids.append(j)
            att_w.append(weight[i][j])
            Xc.append(coords[i][j][0])
            Yc.append(coords[i][j][1])

    pred = y_preds.sigmoid().sum(1).round()
    valid_pred.append(pred.cpu().numpy())
    valid_label.append(label.sum(1).cpu().numpy())

valid_pred = np.concatenate(valid_pred, axis=None)
valid_label = np.concatenate(valid_label, axis=None)

img_ids = np.concatenate(img_ids, axis=None)
tile_ids = np.concatenate(tile_ids, axis=None)
att_w = np.concatenate(att_w, axis=None)
Xc = np.concatenate(Xc, axis=None)
Yc = np.concatenate(Yc, axis=None)

valid_cm = np.array(confusion_matrix(valid_label, valid_pred))

valid_acc = accuracy_score(valid_label, valid_pred)
score = cohen_kappa_score(valid_label, valid_pred, weights='quadratic')

sub_df = pd.DataFrame({'image_id': img_ids, 'tile_id': tile_ids, 'weight': att_w, 'X': Xc, 'Y': Yc})
sub_df.to_csv('../../data/radbound_weight_tkarolinska_0.csv', index=False)

print("Validation Accuracy:", valid_acc)
print("Kappa Score", score)

print(valid_cm)




# for idx,(img,label,_,coords,image_id)  in enumerate(tqdm((validloader))):
#     if train_on_gpu:
#         img, label = img.to(device), label.to(device)
#
#     avg_preds = []
#     for state in states:
#         model.load_state_dict(state)
#         model.to()
#         model.eval()
#         with torch.no_grad():
#             # y_preds1,_,_ = model(img)
#             # y_preds2,_,_ = model(img.flip(-1))
#             # y_preds = y_preds1 + y_preds2
#             y_preds = 0.5*model(img)[0] + 0.5*model(img.flip(-1))[0]
#         avg_preds.append(y_preds.sigmoid().to('cpu').numpy())
#     avg_preds = np.mean(avg_preds, axis=0)

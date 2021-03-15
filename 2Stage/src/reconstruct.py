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
split = 0
N = 6
nfolds = 4
epochs = 20
DEBUG = False


files_path = '../../CLAM/pandaPatches10x/patches'
train_csv = '../../data/train.csv'


# files_path = '../../CLAM/TCGA/patches'
# train_csv = '../../data/prostate_tcga.csv'

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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


###############
# df = pd.read_csv(train_csv,sep='\t').set_index('short_image_id')
# files = sorted(set([p for p in os.listdir(files_path) if p.endswith('.h5')]))
# df['image_id'] = 'New'
# for i in files:
#     x = i[:12]
#     df.at[x,'image_id'] = i[:-3]
#
# # df = df.reset_index().set_index('image_id')
# ## General
# # files = sorted(set([p[:-3] for p in os.listdir(files_path) if p.endswith('.h5')]))
# files = sorted(set([p[:12] for p in os.listdir(files_path) if p.endswith('.h5')]))
# df = df.loc[files]
# df = df.reset_index()

############################
df = pd.read_csv(train_csv).set_index('image_id')

files = sorted(set([p[:-3] for p in os.listdir(files_path) if p.endswith('.h5')]))
df = df.loc[files]
df = df.reset_index()

# df = df[df['isup_grade'] != 0]
df = df[df['data_provider'] == 'radboud']
# df = df[df['data_provider'] == 'karolinska']

splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
splits = list(splits.split(df, df.isup_grade))
folds_splits = np.zeros(len(df)).astype(np.int)

for i in range(nfolds): folds_splits[splits[i][1]] = i
df["split"] = folds_splits

print("Previous Length", len(df))
if DEBUG:
    df = df[:1000]

print("Usable Length", len(df))

"""Mean and Std deviation"""
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

"""Dataset"""

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

df_train = df[df["split"] != split]
df_valid = df[df["split"] == split]

v_dataset = Whole_Slide_Bag(df_valid, files_path, valid_transform, num_patches=N)
validloader = DataLoader(v_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)


"""Training"""
model = AdaptNet(tile_size=size)
# model.load_state_dict(torch.load('../OUTPUT/radboud/adapt/split_3/adapt_128_34_0.0011293814293042357.pth', map_location=torch.device(device)))
model.load_state_dict(torch.load('../OUTPUT/radboud/adapt/split_2/adapt_128_8_0.39798820267121.pth', map_location=torch.device(device)))
# model = nn.DataParallel(model, device_ids=[2,3])
model.to(device)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]

unorm = UnNormalize(mean,std)

import matplotlib.pyplot as plt
def show_image_batch(img_list, title=None, name='foo.png'):
    num = len(img_list)
    fig = plt.figure(figsize=(10,10))
    for i in range(num):
        ax = fig.add_subplot(2, num, i+1)
        ax.imshow(unorm(img_list[i]).numpy().transpose([1,2,0]))
        # ax.imshow(img_list[i].numpy().transpose([1,2,0]))
        ax.set_title(title[i])

    plt.savefig(name)
    plt.show()



testiter = iter(validloader)
imgs,_,_,_,_ = testiter.next()
# imgs,_,_,_,_ = testiter.next()

img = imgs.view(-1, 3, size, size)
with torch.no_grad():
    model.eval()
    r_img = model(img.to(device)).sigmoid()
# val_loss = criterion(r_img, img)
# avg_valid_loss += val_loss.item()

show_image_batch(r_img.cpu(), title=[x for x in range(N)], name='r_img.png')
show_image_batch(img, title=[x for x in range(N)], name='img.png')

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
split = 2
N = 40
nfolds = 4
epochs = 100
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
df = df[df['data_provider'] != 'radboud']
# df = df[df['data_provider'] == 'karolinska']

splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
splits = list(splits.split(df, df.isup_grade))
folds_splits = np.zeros(len(df)).astype(np.int)

for i in range(nfolds): folds_splits[splits[i][1]] = i
df["split"] = folds_splits

print("Previous Length", len(df))
if DEBUG:
    df = df[:500]

print("Usable Length", len(df))

"""Mean and Std deviation"""
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

"""Dataset"""
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor()])
    # transforms.Normalize(mean, std)])

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor()])
    # transforms.Normalize(mean, std)])

df_train = df[df["split"] != split]
df_valid = df[df["split"] == split]

t_dataset = Whole_Slide_Bag(df_train, files_path, train_transform, num_patches=N)
v_dataset = Whole_Slide_Bag(df_valid, files_path, valid_transform, num_patches=N)
print('Length of training and validation set are {} {}'.format(len(t_dataset), len(v_dataset)))

trainloader = DataLoader(t_dataset, batch_size=2*8, shuffle=True, num_workers=4, drop_last=True)
validloader = DataLoader(v_dataset, batch_size=2*8, shuffle=False, num_workers=4, drop_last=True)

writer = SummaryWriter(f'../OUTPUT/radboud/adapt/split_{split}')

"""Training"""
model = AdaptNet(tile_size=size)
model = nn.DataParallel(model)
model.to(device)

# criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss()
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=3e-3, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, div_factor = 10, pct_start=1/epochs, steps_per_epoch=len(trainloader), epochs=epochs)

training_loss = []
validation_loss = []
b_val = np.Inf

print("Training Started for Split:", split)
training_loss = []
validation_loss = []

for epoch in range(epochs):
    start_time = time.time()

    avg_train_loss = 0.0
    l_rate = optimizer.param_groups[0]["lr"]
    model.train()
    for idx,(img,_,_,_,_)  in enumerate(tqdm((trainloader))):
        if train_on_gpu:
            img = img.to(device)

        optimizer.zero_grad()
        r_img = model(img)
        loss = criterion(r_img, img.view(-1,3,size,size))
        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item()
#         print(optimizer.param_groups[0]["lr"])
        scheduler.step()
        # if((idx+1)%(len(trainloader)//5)==0):
            # print('BatchId {}/{} \t train_loss={:.4f}'.format(idx + 1, len(trainloader), avg_train_loss/(idx+1)))

    model.eval()
    avg_valid_loss = 0.0
    with torch.no_grad():
        for img,_,_,_,_ in tqdm(validloader):
            if train_on_gpu:
                img = img.to(device)

            r_img = model(img)
            val_loss = criterion(r_img, img.view(-1,3,size,size))
            avg_valid_loss += val_loss.item()

    # print(avg_train_loss, avg_valid_loss)
    avg_train_loss /= len(trainloader)
    avg_valid_loss /= len(validloader)

    training_loss.append(avg_train_loss)
    validation_loss.append(avg_valid_loss)
#     l_rate = optimizer.param_groups[0]["lr"]

    writer.add_scalars('Loss', {'Training Loss': avg_train_loss, 'Validation Loss': avg_valid_loss}, epoch)
    writer.add_scalar('Learning Rate', l_rate , epoch)

    if(b_val>avg_valid_loss):
        torch.save(model.module.state_dict(), "../OUTPUT/radboud/adapt/split_{}/adapt_{}_{}_{}.pth".format(split, size, epoch+1, avg_valid_loss))
        b_val = avg_valid_loss

#     scheduler.step(avg_valid_loss)
    # scheduler.step()
    time_taken = time.time() - start_time
    print('Epoch {}/{} \t train_loss={:.4f} \t valid_loss={:.4f} \t l_rate={:.8f} \t time={:.2f}s'.\
          format(epoch + 1, epochs, avg_train_loss, avg_valid_loss, l_rate, time_taken))

writer.close()

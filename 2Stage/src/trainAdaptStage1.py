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
split = 3
N = 64
nfolds = 4
epochs = 20
DEBUG = True


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

df = df[df['isup_grade'] != 0]
df = df[df['data_provider'] != 'karolinska']

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
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

"""Dataset"""
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


for split in range(3,nfolds):

    df_train = df[df["split"] != split]
    df_valid = df[df["split"] == split]

    t_dataset = Whole_Slide_Bag(df_train, files_path, train_transform, num_patches=N)
    v_dataset = Whole_Slide_Bag(df_valid, files_path, valid_transform, num_patches=N)
    print('Length of training and validation set are {} {}'.format(len(t_dataset), len(v_dataset)))

    trainloader = DataLoader(t_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    validloader = DataLoader(v_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)

    writer = SummaryWriter(f'../OUTPUT/radboud/stage1/split_{split}')

    """Training"""
    model = EfficientModel(c_out=4, tile_size=size, n_tiles=N)
    # model = nn.DataParallel(model)
    model.to(device)

    net1 = AdaptNet(tile_size=size)
    net1.load_state_dict(torch.load('../OUTPUT/radboud/adapt/split_3/adapt_128_34_0.0011293814293042357.pth', map_location=torch.device(device)))
    net1 = nn.DataParallel(net1)
    net1.to(device)


    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, div_factor = 10, pct_start=1/epochs, steps_per_epoch=len(trainloader), epochs=epochs)

    training_loss = []
    validation_loss = []
    k_score = 0.0

    print("Training Started for Split:", split)
    training_loss = []
    validation_loss = []
    k_score = 0.0

    for epoch in range(epochs):
        start_time = time.time()

        train_pred = []
        valid_pred = []
        train_label = []
        valid_label = []
        avg_train_loss = 0.0
        avg_instance_loss = 0.0
        l_rate = optimizer.param_groups[0]["lr"]
        model.train()
        for idx,(img,label,_,_,_)  in enumerate(tqdm((trainloader))):
            if train_on_gpu:
                img, label = img.to(device), label.to(device)

            with torch.no_grad():
                img = img.view(-1,3,size,size)
                img = net1(img)
                img = img.view(-1,N,3,size,size)

            # print(img.shape)
            label = label.long()
            optimizer.zero_grad()
            logits, _, instance_loss = model(img)
            loss = criterion(logits.float(), label.float())
            t_loss = 0.7*loss + 0.3*instance_loss
            t_loss.backward()
            optimizer.step()

            pred = logits.sigmoid().sum(1).detach().round()
            train_pred.append(pred.cpu())
            train_label.append(label.sum(1).cpu())

            avg_train_loss += loss.item()
            avg_instance_loss += instance_loss.item()
    #         print(optimizer.param_groups[0]["lr"])
            scheduler.step()
            if((idx+1)%(len(trainloader)//10)==0):
                print('BatchId {}/{} \t train_loss={:.4f} \t instance_loss={:.4f} '.format(idx + 1, len(trainloader), avg_train_loss/(idx+1), avg_instance_loss/(idx+1)))

                # print('BatchId {}/{} \t train_loss={:.4f} \t instance_loss={:.4f} \t train_kappa={:.4f} \t train_acc={:.4f}'.format(idx + 1, len(trainloader), avg_train_loss/(idx+1), avg_instance_loss/(idx+1), cohen_kappa_score(train_label, train_pred, weights='quadratic'), accuracy_score(train_label, train_pred)))

        model.eval()
        avg_valid_loss = 0.0
        with torch.no_grad():
            for img,label,_,_,_ in tqdm(validloader):
                if train_on_gpu:
                    img, label = img.to(device), label.to(device)

                with torch.no_grad():
                    img = img.view(-1,3,size,size)
                    img = net1(img)
                    img = img.view(-1,N,3,size,size)

                logits1, _, instance_loss1 = model(img)
                logits2, _, instance_loss2 = model(img.flip(-1))

                logits = 0.5*logits1 + 0.5*logits2
                instance_loss = 0.5*instance_loss1 + 0.5*instance_loss2

                val_loss = criterion(logits.float(), label.float())
                v_loss = 0.7*val_loss + 0.3*instance_loss
                avg_valid_loss += v_loss.item()

                pred = logits.sigmoid().sum(1).detach().round()
                valid_pred.append(pred.cpu())
                valid_label.append(label.sum(1).cpu())


        train_pred = torch.cat(train_pred).cpu().numpy()
        train_label = torch.cat(train_label).cpu().numpy()
        valid_pred = torch.cat(valid_pred).cpu().numpy()
        valid_label = torch.cat(valid_label).cpu().numpy()

        train_cm = np.array(confusion_matrix(train_label, train_pred))
        valid_cm = np.array(confusion_matrix(valid_label, valid_pred))

        avg_train_loss /= len(trainloader)
        avg_valid_loss /= len(validloader)
        avg_instance_loss /= len(trainloader)
        train_acc = accuracy_score(train_label, train_pred)
        valid_acc = accuracy_score(valid_label, valid_pred)
        score = cohen_kappa_score(valid_label, valid_pred, weights='quadratic')

        training_loss.append(avg_train_loss)
        validation_loss.append(avg_valid_loss)
    #     l_rate = optimizer.param_groups[0]["lr"]

        writer.add_scalar('Valid Kappa Score', score , epoch)
        writer.add_scalars('Accuracy', {'Training Accuracy': train_acc,'Validation Accuracy': valid_acc}, epoch)
        writer.add_scalars('Loss', {'Training Loss': avg_train_loss, 'Training Instance Loss': avg_instance_loss, 'Validation Loss': avg_valid_loss}, epoch)
        writer.add_scalar('Learning Rate', l_rate , epoch)

        if(k_score<score):
            torch.save(model.state_dict(), "../OUTPUT/radboud/stage1/split_{}/efficient_b0_{}_{}.pth".format(split, epoch+1, score))
            np.savetxt(f'../OUTPUT/radboud/stage1/split_{split}/valid_cm_{epoch+1}_{score}.txt', valid_cm, fmt='%10.0f')
            np.savetxt(f'../OUTPUT/radboud/stage1/split_{split}/train_cm_{epoch+1}_{score}.txt', train_cm, fmt='%10.0f')
            k_score = score

    #     scheduler.step(avg_valid_loss)
        # scheduler.step()
        time_taken = time.time() - start_time
        print("Train CM:")
        print(train_cm)
        print("Valid CM:")
        print(valid_cm)

        print('Epoch {}/{} \t train_loss={:.4f} \t valid_loss={:.4f} \t train_acc={:.4f} \t valid_acc={:.4f} \t valid_kappa={:.4f}  \t l_rate={:.8f} \t time={:.2f}s'.\
              format(epoch + 1, epochs, avg_train_loss, avg_valid_loss, train_acc, valid_acc, score, l_rate, time_taken))

    writer.close()

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
N = 20
nfolds = 4
epochs = 20
DEBUG = True


# files_path = '../../CLAM/pandaPatches10x/patches'
# train_csv = '../../data/train.csv'


files_path = '../../CLAM/TCGA/patches'
train_csv = '../../data/prostate_tcga.csv'

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
df = pd.read_csv(train_csv,sep='\t').set_index('short_image_id')
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

############################
# df = pd.read_csv(train_csv).set_index('image_id')
#
# files = sorted(set([p[:-3] for p in os.listdir(files_path) if p.endswith('.h5')]))
# df = df.loc[files]
# df = df.reset_index()
#
# # df = df[df['isup_grade'] != 0]
# df = df[df['data_provider'] != 'radboud']
# df = df[df['data_provider'] == 'karolinska']

print("Previous Length", len(df))
if DEBUG:
    df = df[:30]

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


v_dataset = Whole_Slide_Bag(df, files_path, valid_transform, num_patches=N)
validloader_tcga = DataLoader(v_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)


####################################### Radboud

files_path = '../../CLAM/pandaPatches10x/patches'
train_csv = '../../data/train.csv'

df = pd.read_csv(train_csv).set_index('image_id')

files = sorted(set([p[:-3] for p in os.listdir(files_path) if p.endswith('.h5')]))
df = df.loc[files]
df = df.reset_index()

# df = df[df['isup_grade'] != 0]
df = df[df['data_provider'] == 'radboud']
# df = df[df['data_provider'] == 'karolinska']

print("Previous Length", len(df))
if DEBUG:
    df = df[:30]

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

v_dataset = Whole_Slide_Bag(df, files_path, valid_transform, num_patches=N)
validloader_rad = DataLoader(v_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)


###################################### Karolinska
files_path = '../../CLAM/pandaPatches10x/patches'
train_csv = '../../data/train.csv'

df = pd.read_csv(train_csv).set_index('image_id')

files = sorted(set([p[:-3] for p in os.listdir(files_path) if p.endswith('.h5')]))
df = df.loc[files]
df = df.reset_index()

# df = df[df['isup_grade'] != 0]
# df = df[df['data_provider'] == 'radboud']
df = df[df['data_provider'] == 'karolinska']

print("Previous Length", len(df))
if DEBUG:
    df = df[:30]

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

v_dataset = Whole_Slide_Bag(df, files_path, valid_transform, num_patches=N)
validloader_kar = DataLoader(v_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)


"""Training"""
model = AdaptNet(tile_size=size)
model.load_state_dict(torch.load('../OUTPUT/radboud/adapt/split_3/adapt_128_50_0.000691256259978261.pth', map_location=torch.device(device)))
# model = nn.DataParallel(model, device_ids=[2,3])
model.to(device)

# from efficientnet_pytorch import EfficientNet
# name = 'efficientnet-b0'
# m = EfficientNet.from_pretrained(name)
# c_feature = m._fc.in_features
# m._fc = nn.Identity()
# m.to(device)
# m.eval()


m = EfficientModel(c_out=4, tile_size=size, n_tiles=N)
# m.load_state_dict(torch.load('../OUTPUT/tcga/stage1/split_0/efficient_b0_20_0.6779463243873979.pth', map_location=torch.device(device)))
# # m.load_state_dict(torch.load('../OUTPUT/radbound/stage1/split_0/efficient_b0_17_0.6364211353668056.pth', map_location=torch.device(device)))
m.load_state_dict(torch.load('../OUTPUT/radboud/stage1/split_3/efficient_b0_12_0.7296633941093968.pth', map_location=torch.device(device)))
m.to(device)
m.eval()

features = []
labels = []


# for imgs,_,_,_,_ in tqdm(validloader_tcga):
#     with torch.no_grad():
#         f = m.feature_extractor(imgs.view(-1,3,size,size).to(device)).cpu()
#         for i in f:
#             features.append(i.unsqueeze(0).numpy())
#             labels.append('tcga')

# for imgs,_,_,_,_ in tqdm(validloader_kar):
#     with torch.no_grad():
#         f = m.feature_extractor(imgs.view(-1,3,size,size).to(device)).cpu()
#         for i in f:
#             features.append(i.unsqueeze(0).numpy())
#             labels.append('karolinska')
#
# for imgs,_,_,_,_ in tqdm(validloader_rad):
#     with torch.no_grad():
#         f = m.feature_extractor(imgs.view(-1,3,size,size).to(device)).cpu()
#         for i in f:
#             features.append(i.unsqueeze(0).numpy())
#             labels.append('radboud')
#
# pooling = nn.AdaptiveAvgPool2d((1,1))
# def feat(x):
#     x = F.relu(model.conv1(x))
#     x = F.relu(model.conv2(x))
#     x = F.relu(model.conv3(x))
#     x = pooling(x)
#
#     return x

# for imgs,_,_,_,_ in tqdm(validloader_tcga):
#     with torch.no_grad():
#         x = model(imgs.view(-1,3,size,size).to(device))
#         f = feat(x).squeeze().cpu()
#         # f = m(x).cpu()
#         for i in f:
#             features.append(i.unsqueeze(0).numpy())
#             labels.append('tcga')
#
# for imgs,_,_,_,_ in tqdm(validloader_kar):
#     with torch.no_grad():
#         x = model(imgs.view(-1,3,size,size).to(device))
#         f = feat(x).squeeze().cpu()
#         for i in f:
#             features.append(i.unsqueeze(0).numpy())
#             labels.append('karolinska')
#
# for imgs,_,_,_,_ in tqdm(validloader_rad):
#     with torch.no_grad():
#         x = model(imgs.view(-1,3,size,size).to(device))
#         f = feat(x).squeeze().cpu()
#         for i in f:
#             features.append(i.unsqueeze(0).numpy())
#             labels.append('radboud')



for imgs,_,_,_,_ in tqdm(validloader_tcga):
    with torch.no_grad():
        x = model(imgs.view(-1,3,size,size).to(device))
        f = m.feature_extractor(x).cpu()
        for i in f:
            features.append(i.unsqueeze(0).numpy())
            labels.append('tcga')

for imgs,_,_,_,_ in tqdm(validloader_kar):
    with torch.no_grad():
        x = model(imgs.view(-1,3,size,size).to(device))
        f = m.feature_extractor(x).cpu()
        for i in f:
            features.append(i.unsqueeze(0).numpy())
            labels.append('karolinska')

for imgs,_,_,_,_ in tqdm(validloader_rad):
    with torch.no_grad():
        x = model(imgs.view(-1,3,size,size).to(device))
        f = m.feature_extractor(x).cpu()
        for i in f:
            features.append(i.unsqueeze(0).numpy())
            labels.append('radboud')
#


features = np.concatenate(features, axis = 0)
print(features.shape)


import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE


tsne = TSNE()
tsne_results = tsne.fit_transform(features)


fig = plt.figure(figsize=(6,4))
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=labels,
    legend="full",
    s=10,
)

plt.title('t-SNE Features plot', fontweight='bold', fontsize=8)
plt.savefig('AdaptCNN-tsne.png', bbox_inches='tight', dpi=200)
# plt.savefig('CNN-tsne.png', bbox_inches='tight', dpi=200)



#
# class UnNormalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
#             # The normalize code -> t.sub_(m).div_(s)
#         return tensor
#
# # mean=[0.485, 0.456, 0.406]
# # std=[0.229, 0.224, 0.225]
#
# unorm = UnNormalize(mean,std)
#
# import matplotlib.pyplot as plt
# def show_image_batch(img_list, title=None, name='foo.png'):
#     num = len(img_list)
#     fig = plt.figure(figsize=(10,10))
#     for i in range(num):
#         ax = fig.add_subplot(2, num, i+1)
#         ax.imshow(unorm(img_list[i]).numpy().transpose([1,2,0]))
#         # ax.imshow(img_list[i].numpy().transpose([1,2,0]))
#         ax.set_title(title[i])
#
#     plt.savefig(name)
#     plt.show()
#
#
#
# testiter = iter(validloader)
# imgs,_,_,_,_ = testiter.next()
# # imgs,_,_,_,_ = testiter.next()
#
# img = imgs.view(-1, 3, size, size)
# with torch.no_grad():
#     model.eval()
#     r_img = model(img.to(device))
# # val_loss = criterion(r_img, img)
# # avg_valid_loss += val_loss.item()
#
# show_image_batch(r_img.cpu(), title=[x for x in range(5)], name='r_img.png')
# show_image_batch(img, title=[x for x in range(5)], name='img.png')

from dataset.dataset import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

size = 128

# df = pd.read_csv('../../data/train.csv').set_index('image_id')
# files_path = '../../CLAM/pandaPatches10x/patches'


files_path = '../../CLAM/TCGA/patches'
train_csv = '../../data/prostate_tcga.csv'
weight = '../../data/tcga_weight_tkarolinska_0.csv'


## TCGA
df = pd.read_csv(train_csv,sep='\t').set_index('short_image_id')
files = sorted(set([p for p in os.listdir(files_path) if p.endswith('.h5')]))
df['image_id'] = 'New'
for i in files:
    x = i[:12]
    df.at[x,'image_id'] = i[:-3]


# print(df['image_id'][0])


c = 0
for i in files:
    # print(i[:7])
    if(i[:7] == 'TCGA-HC'):
        c += 1
        print(i)

print(c)






# files = sorted(set([p[:-3] for p in os.listdir(files_path) if p.endswith('.h5')]))
# df = df.loc[files]
# df = df.reset_index()
#
# mean = torch.tensor([1.0-0.87715868, 1.0-0.75073279, 1.0-0.83313599])
# std = torch.tensor([0.39358389, 0.52248967, 0.42447533])
#
# train_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(size),
#     transforms.RandomRotation(45),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)])
#
# valid_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)])
#
# t_dataset = Whole_Slide_Bag(df,files_path,train_transform)
# trainloader = DataLoader(t_dataset, batch_size=1, shuffle=True, num_workers=1)
#
# # c = 0
# # from tqdm import tqdm
# # for i,j,k,l in tqdm(trainloader):
# #     i = i.squeeze()
# #     print(i.shape)
#
#
# def blue_ratio(imgs):
#     allR = []
#     for i in range(len(imgs)):
#         B = imgs[i][:,:,0].mean()
#         G = imgs[i][:,:,1].mean()
#         R = imgs[i][:,:,2].mean()
#
#         ratio = ((100 * B)/(1+R+G)) * (256./(1+B+R+G))
#         allR.append(ratio)
#
#     return np.array(allR)
#
# import h5py
# import cv2
# file_path = os.path.join(files_path,df.image_id.values[0] + '.h5')
# with h5py.File(file_path,'r') as hdf5_file:
#     imgs = np.array(hdf5_file['imgs']) #BGR
#     coords = np.array(hdf5_file['coords'])
#
#     index = np.argsort(-blue_ratio(imgs))
#     # print(x)
#     # print(coords)
#
#     # img = cv2.cvtColor(imgs[9], cv2.COLOR_RGB2BGR)
#     # cv2.imwrite('../color_img_1.jpg', img)
#
#     # print(index)
#     # print(coords)
#     imgs = imgs[index]
#     coords = coords[index]
#
#
#
# import os
# import sys
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
#
# import numpy as np
# import pandas as pd
# import random
#
# import PIL
# from PIL import Image
# import skimage.io
# from tqdm import tqdm
# import time
#
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# import torch.nn.functional as F
#
# from dataset.dataset import *
# from net.modules import *
#
# from torch.utils.tensorboard import SummaryWriter
#
#
# SEED = 2021
# size = 128
# # split = 0
# nfolds = 4
# epochs = 20
# DEBUG = False
#
#
# files_path = '../../CLAM/pandaPatches10x/patches'
# train_csv = '../../data/train.csv'
#
#
# def create_positive_targets(length, device):
#     return torch.full((length, ), 1, device=device).long()
# def create_negative_targets(length, device):
#     return torch.full((length, ), 0, device=device).long()
#
# from sklearn.utils import shuffle
#
# def inst_eval(w, input):
#     device = input.device
#     w = w.squeeze()
#     input = input.squeeze(0)
#
#     top_p_ids = torch.topk(w, 16)[1]
#     top_n_ids = torch.topk(-w, 16)[1]
#
#     top_p, top_n, p_targets, n_targets = [],[],[],[]
#     for i in range(w.shape[0]):
#         top_p.append(torch.index_select(input[i], dim=0, index=top_p_ids[i]))
#         top_n.append(torch.index_select(input[i], dim=0, index=top_n_ids[i]))
#         p_targets.append(create_positive_targets(16, device))
#         n_targets.append(create_negative_targets(16, device))
#
#
#     all_targets = torch.cat([torch.cat(p_targets), torch.cat(n_targets)])
#     all_instances = torch.cat([torch.cat(top_p), torch.cat(top_n)])
#
#     all_instances,all_targets = shuffle(all_instances,all_targets)
#
#     print(all_instances.shape)
#     print(all_targets.shape)
#     print(all_targets)
#
#
#
# x = os.listdir('../../data/panda/train_images')
# id = [i[:32] for i in x]
# #
# # df = pd.read_csv('../../data/train.csv').set_index('image_id')
#
# df2  = pd.read_csv('../../data/tcga_weight_tkarolinska_0.csv').set_index('image_id')
# df = df.loc[id]
#
# print(df)
# t = df2.loc['TCGA-2A-A8VL-01Z-00-DX1.2C2BD6EF-EC17-4117-AE89-A22B67AFB233']
#
# w = t.weight.values
# x = t.X.values
# y = t.Y.values
#
# print(len(w))
#
#
# coord = []
# for i in range(len(w)):
#     coord.append([4*x[i],4*y[i]])
#
# file_path = '../../data/panda/train_images/cb88fb1e0305ba94d8186a245ac2193b.tiff'
# print(os.path.exists(file_path))
#
#
# import openslide
# wsi = openslide.open_slide(path)
#
# region_size = wsi.level_dimensions[1]
# top_left = (0,0)
# bot_right = self.level_dim[0]
# w, h = region_size
#
#
# overlay = np.full(np.flip(region_size), 0).astype(float)
# counter = np.full(np.flip(region_size), 0).astype(np.uint16)
# count = 0
# for idx in range(len(coords)):
#     score = scores[idx]
#     coord = coords[idx]
#     if score >= threshold:
#         if binarize:
#             score=1.0
#             count+=1
#     else:
#         score=0.0
#     # accumulate attention
#     overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
#     # accumulate counter
#     counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1
#









# import sys
# sys.path = [
#     '../../CLAM',
# ] + sys.path

# from wsi_core.WholeSlideImage import WholeSlideImage
#
# ws = WholeSlideImage(file_path)
#
#
# # print(np.array(coord).shape)
# heatmap = ws.visHeatmap(np.array(w), np.array(coord), patch_size=(256*4, 256*4), segment=False, cmap='terrain')
# from PIL import Image
#
# heatmap.save('heatmpa.png')




# print(t)






# w = torch.randn(4,64)
# h = torch.randn(4,64,1024)
#
# inst_eval(w,h)

# for i,_,_,_,_ in validloader:
#     print(i.shape)






    # index = np.argsort(-blue_ratio(imgs))

    # print(index)
    # for i in range(len(imgs)):
    #     img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f'../color_img_{i}.jpg', img)


    # print(len(imgs))




    # """Training"""
    # model = EfficientModel(c_out=5, tile_size=size)
    # # model = nn.DataParallel(model, device_ids=[2,3])
    # model.to(device)
    #
    # # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(),lr=3e-4, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, div_factor = 10, pct_start=1/epochs, steps_per_epoch=len(trainloader), epochs=epochs)

    # training_loss = []
    # validation_loss = []
    # k_score = 0.0
    #
    # training_loss = []
    # validation_loss = []
    # k_score = 0.0
    #
    # for epoch in range(epochs):
    #     start_time = time.time()
    #
    #     train_pred = []
    #     valid_pred = []
    #     train_label = []
    #     valid_label = []
    #     avg_train_loss = 0.0
    #     avg_instance_loss = 0.0
    #     l_rate = optimizer.param_groups[0]["lr"]
    #     model.train()
    #     for idx,(img,_,label,_) in enumerate(tqdm(trainloader)):
    #         if train_on_gpu:
    #             img, label = img.to(device), label.to(device)
    #
    #         # label = label.long()
    #         optimizer.zero_grad()
    #         logits, _, instance_loss = model(img)
    #         loss = criterion(logits, label.squeeze(1))
    #         t_loss = loss + 0.03*instance_loss
    #
    #         t_loss.backward()
    #         optimizer.step()
    #
    #         # pred = logits.sigmoid().sum(1).detach().round()
    #         _,pred = logits.topk(1,dim=1)
    #         train_pred.append(pred.cpu())
    #         train_label.append(label.cpu())
    #
    #         avg_train_loss += loss.item()
    #         avg_instance_loss += instance_loss.item()
    # #         print(optimizer.param_groups[0]["lr"])
    #         scheduler.step()
    #
    #         if((idx+1)%100==0):
    #             print('BatchId {}/{} \t train_loss={:.4f} \t instance_loss={:.4f} \t train_kappa={:.4f} \t train_acc={:.4f}'.format(idx + 1, len(trainloader), avg_train_loss/(idx+1), avg_instance_loss/(idx+1), cohen_kappa_score(train_label, train_pred, weights='quadratic'), accuracy_score(train_label, train_pred)))
    #
    #     model.eval()
    #     avg_valid_loss = 0.0
    #     with torch.no_grad():
    #         for img,_,label,_ in tqdm(validloader):
    #             if train_on_gpu:
    #                 img, label = img.to(device), label.to(device)
    #
    #             logits, _,instance_loss = model.forward(img)
    #
    #             val_loss = criterion(logits, label.squeeze(1))
    #             avg_valid_loss += val_loss.item()
    #
    #             # pred = logits.sigmoid().sum(1).detach().round()
    #             _,pred = logits.topk(1,dim=1)
    #             valid_pred.append(pred)
    #             valid_label.append(label.cpu())
    #
    #     train_pred = torch.cat(train_pred).cpu().numpy()
    #     train_label = torch.cat(train_label).cpu().numpy()
    #     valid_pred = torch.cat(valid_pred).cpu().numpy()
    #     valid_label = torch.cat(valid_label).cpu().numpy()
    #
    #     train_cm = np.array(confusion_matrix(train_label, train_pred))
    #     valid_cm = np.array(confusion_matrix(valid_label, valid_pred))
    #
    #     avg_train_loss /= len(trainloader)
    #     avg_valid_loss /= len(validloader)
    #     train_acc = accuracy_score(train_label, train_pred)
    #     valid_acc = accuracy_score(valid_label, valid_pred)
    #     score = cohen_kappa_score(valid_label, valid_pred, weights='quadratic')
    #
    #     training_loss.append(avg_train_loss)
    #     validation_loss.append(avg_valid_loss)
    #     print("Train CM:", train_cm)
    #     print("Valid CM:", valid_cm)
    # #     l_rate = optimizer.param_groups[0]["lr"]
    #
    # #     writer.add_scalar('Valid Kappa Score', score , epoch)
    # #     writer.add_scalars('Accuracy', {'Training Accuracy': train_acc,'Validation Accuracy': valid_acc}, epoch)
    # #     writer.add_scalars('Loss', {'Training Loss': avg_train_loss,'Validation Loss': avg_valid_loss}, epoch)
    # #     writer.add_scalar('Learning Rate', l_rate , epoch)
    #
    #     # if(k_score<score):
    #     #     torch.save(model.module.state_dict(), "stage1/split_{}/{}/{}_1_{}.pth".format(split, model_name, model_name, score))
    #     #     np.savetxt(f'stage1/split_{split}/{model_name}/valid_cm_{epoch+1}_{score}.txt', valid_cm, fmt='%10.0f')
    #     #     np.savetxt(f'stage1/split_{split}/{model_name}/train_cm_{epoch+1}_{score}.txt', train_cm, fmt='%10.0f')
    #     #     k_score = score
    #
    # #     scheduler.step(avg_valid_loss)
    #     # scheduler.step()
    #     time_taken = time.time() - start_time
    #
    #     print('Epoch {}/{} \t train_loss={:.4f} \t valid_loss={:.4f} \t train_acc={:.4f} \t valid_acc={:.4f} \t valid_kappa={:.4f}  \t l_rate={:.8f} \t time={:.2f}s'.\
    #           format(epoch + 1, epochs, avg_train_loss, avg_valid_loss, train_acc, valid_acc, score, l_rate, time_taken))

    # writer.close()

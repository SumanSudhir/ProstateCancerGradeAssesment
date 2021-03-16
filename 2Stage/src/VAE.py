import pandas as pd
import numpy as np

import torch
import os
import time
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from itertools import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from net.modules import *
from dataset.dataset import *
from utils.utils import *

class VAE(object):
    def __init__(self, args):
        self.gpu = args.gpu
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.slope = args.slope

        self.z_dim = args.z_dim
        self.h_dim = args.h_dim
        self.learning_rate = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.decay = args.decay
        self.X_dim = 128*128*3

        self.split = args.split
        self.csv_path = args.csv_path
        self.files_path = args.files_path
        self.result_dir = args.result_dir
        self.save_dir = args.save_dir

        self.Net = VAEncoder()
        self.unorm = UnNormalize()

        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=self.learning_rate, betas=[self.beta1,self.beta2], weight_decay=self.decay)

        if self.gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.Net.to(self.device)

        print('---------- Networks architecture -------------')
        print_network(self.Net)
        print('-----------------------------------------------')

        df = pd.read_csv(self.csv_path)
        df_train = df[df['split'] != self.split]
        df_valid = df[df['split'] == self.split]

        t_dataset = Whole_Slide_Tiles(df_train, self.files_path, custom_transforms=get_transform(data='train'))
        v_dataset = Whole_Slide_Tiles(df_valid, self.files_path, custom_transforms=get_transform(data='valid'))

        self.trainloader = DataLoader(t_dataset, self.batch_size, shuffle=True, num_workers=4)
        self.validloader = DataLoader(v_dataset, self.batch_size, shuffle=True, num_workers=4)

    def reset_grad(self):
        self.Net.zero_grad()

    def train(self):
        self.train_hist = {}
        self.train_hist['loss'] = []

        self.eval_hist = {}
        self.eval_hist['loss'] = []
        self.eval_hist['pixel_norm'] = []
        self.eval_hist['z_norm'] = []

        for epoch in range(self.epochs):
            self.Net.train()

            train_loss = 0.0

            for batchID, (data, image_id) in enumerate(tqdm(self.trainloader)):
                z = torch.randn(data.shape[0], self.z_dim, 1, 1, requires_grad=True)
                X = data

                if self.gpu:
                    X, z = X.to(self.device), z.to(self.device)

                self.optimizer.zero_grad()
                X_hat, mu, logvar = self.Net(X)

                loss = VAE_loss(X_hat, X, mu, logvar)

                loss.backward()
                self.optimizer.step()
                # self.reset_grad()

                train_loss += loss.item()

                if((1+batchID)%30 == 0):
                    samples = self.unorm(X_hat.detach()).cpu().numpy()

                    fig = plt.figure(figsize=(8,4))
                    gs = gridspec.GridSpec(4,8)
                    gs.update(wspace=0.05, hspace=0.05)

                    for i, sample in enumerate(samples):
                        if i<32:
                            ax = plt.subplot(gs[i])
                            plt.axis('off')
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            ax.set_aspect('equal')

                            sample = np.clip(sample, 0, 1)
                            # sample = sample.reshape(128,128,3)
                            sample = sample.transpose([1,2,0])
                            # sample = np.rot90(sample, 2)
                            plt.imshow(sample)

                    if not os.path.exists(self.result_dir + '/train/'):
                        os.makedirs(self.result_dir + '/train/')

                    filename = "epoch_" + str(epoch) + "_batchid_" + str(batchID+1)
                    plt.savefig(self.result_dir + '/train/{}.png'.format(filename, bbox_inches='tight'))
                    plt.close()

            print("Train loss:", train_loss / len(self.trainloader))

            self.train_hist['loss'].append(train_loss / len(self.trainloader))

    #         # Validation
            self.Net.eval()
    #
            valid_loss = 0
    #
            mean_pixel_norm = 0.0
            mean_z_norm = 0
            norm_counter = len(self.validloader)

            for batchID, (data, image_id) in enumerate(tqdm(self.validloader)):
                z = torch.randn(data.shape[0], self.z_dim, 1, 1)
                X = data
                if self.gpu:
                    X, z = data.to(self.device), z.to(self.device)

                X_hat, mu, logvar = self.Net(X)

                loss = VAE_loss(X_hat, X, mu, logvar)
    #
                valid_loss += loss.item()

                pixel_norm = X-X_hat
                pixel_norm = pixel_norm.norm().item() / float(self.X_dim)
                mean_pixel_norm += pixel_norm

                # norm_counter += 1
            print("Eval loss:", valid_loss/ norm_counter)

            self.eval_hist['loss'].append(valid_loss / norm_counter)

            print("Pixel norm:", mean_pixel_norm / norm_counter)
            self.eval_hist['pixel_norm'].append( mean_pixel_norm / norm_counter )

            with open('pixel_error_BIGAN.txt', 'a') as f:
                f.writelines(str(mean_pixel_norm / norm_counter) + '\n')

            ### Save X and its reconstruction at the end of each epoch
            samples = self.unorm(X.detach()).cpu().numpy()

            fig = plt.figure(figsize=(10,2))
            gs = gridspec.GridSpec(2,10)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                if i<10:
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')

                    sample = np.clip(sample, 0, 1)
                    sample = sample.transpose([1,2,0])
    #                 sample = sample.reshape(128,128,3)
                    # sample = np.rot90(sample, 2)
                    plt.imshow(sample)


            X_hat = self.Net(X)[0]
            samples = self.unorm(X_hat.detach()).cpu().numpy()
            for i, sample in enumerate(samples):
                if i<10:
                    ax = plt.subplot(gs[10+i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
    #
                    sample = np.clip(sample, 0, 1)
                    sample = sample.transpose([1,2,0])
                    # sample = sample.reshape(128,128,3)
                    # sample = np.rot90(sample, 2)
                    plt.imshow(sample)

            if not os.path.exists(self.result_dir + '/recons/'):
                os.makedirs(self.result_dir + '/recons/')

            filename = "epoch_" + str(epoch)
            plt.savefig(self.result_dir + '/recons/{}.png'.format(filename), bbox_inches='tight')
            plt.close()
            self.save_model()

    def save_model(self):
        torch.save(self.Net.state_dict(), self.save_dir + "/VAE.pt")

    # def plot_tsne(self):
    #     df = pd.read_csv('../../data/train.csv')
    #     files = sorted(set([p[:-3] for p in os.listdir(self.files_path) if p.endswith('.h5')]))
    #     df = df.loc[files]
    #     df = df.reset_index()
    #
    #     df_rad = df[df['data_provider'] == 'radboud']
    #     df_kar = df[df['data_provider'] == 'karolinska']
    #
    #     t_dataset = Whole_Slide_Tiles(df_rad, self.files_path, custom_transforms=get_transform(data='train'))
    #     v_dataset = Whole_Slide_Tiles(df_valid, self.files_path, custom_transforms=get_transform(data='valid'))

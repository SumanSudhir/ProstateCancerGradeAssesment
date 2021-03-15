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



class BiGAN(object):
    """
    BiGAN network haveing multiple function to train and validate
    """
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

        params = {'slope':self.slope, 'dropout':self.dropout, 'batch_size':self.batch_size}

        self.G = Generator(self.z_dim, params)
        self.D = Discriminator(self.z_dim, self.h_dim, params)
        self.E = Encoder(self.z_dim, params)

        self.G_optimizer = torch.optim.Adam(chain(self.E.parameters(), self.G.parameters()), lr=self.learning_rate, betas=[self.beta1,self.beta2], weight_decay=self.decay)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=[self.beta1,self.beta2], weight_decay=self.decay)

        if self.gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.G.to(self.device)
            self.D.to(self.device)
            self.E.to(self.device)

        print('---------- Networks architecture -------------')
        print_network(self.G)
        print_network(self.E)
        print_network(self.D)
        print('-----------------------------------------------')

        df = pd.read_csv(self.csv_path)
        df_train = df[df['split'] != self.split]
        df_valid = df[df['split'] == self.split]

        t_dataset = Whole_Slide_Tiles(df_train, self.files_path, custom_transforms=get_transform(data='train'))
        v_dataset = Whole_Slide_Tiles(df_valid, self.files_path, custom_transforms=get_transform(data='valid'))

        self.trainloader = DataLoader(t_dataset, self.batch_size, shuffle=True, num_workers=4)
        self.validloader = DataLoader(v_dataset, self.batch_size, shuffle=True, num_workers=4)

    def reset_grad(self):
        self.G.zero_grad()
        self.D.zero_grad()
        self.E.zero_grad()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []

        self.eval_hist = {}
        self.eval_hist['D_loss'] = []
        self.eval_hist['G_loss'] = []
        self.eval_hist['pixel_norm'] = []
        self.eval_hist['z_norm'] = []

        for epoch in range(self.epochs):
            self.G.train()
            self.D.train()
            self.E.train()

            train_loss_G = 0.0
            train_loss_D = 0.0

            for batchID, (data, image_id) in enumerate(tqdm(self.trainloader)):
                z = torch.randn(data.shape[0], self.z_dim, 1, 1, requires_grad=True)
                X = data

                if self.gpu:
                    X, z = X.to(self.device), z.to(self.device)

                z_hat = self.E(X)
                X_hat = self.G(z)

                D_enc = self.D(X, z_hat)
                D_gen = self.D(X_hat,z)

                D_loss = -torch.mean(log(D_enc) + log(1-D_gen))
                G_loss = -torch.mean(log(D_gen) + log(1-D_enc))

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()
                self.reset_grad()

                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()
                self.reset_grad()

                train_loss_D += D_loss.item()
                train_loss_G += G_loss.item()
                # print(train_loss_D, train_loss_G)


                if(batchID%1000 == 0):
                    samples = X_hat.detach().cpu().numpy()

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
                            sample = sample.reshape(128,128,3)
                            sample = np.rot90(sample, 2)
                            plt.imshow(sample)

                    if not os.path.exists(self.result_dir + '/train/'):
                        os.makedirs(self.result_dir + '/train/')

                    filename = "epoch_" + str(epoch) + "_batchid_" + str(batchID)
                    plt.savefig(self.result_dir + '/train/{}.png'.format(filename, bbox_inches='tight'))
                    plt.close()

            print("Train loss G:", train_loss_G / len(self.trainloader))
            print("Train loss D:", train_loss_D / len(self.trainloader))

            self.train_hist['D_loss'].append(train_loss_D / len(self.trainloader))
            self.train_hist['G_loss'].append(train_loss_G / len(self.trainloader))

            # Validation
            self.G.eval()
            self.D.eval()
            self.E.eval()

            valid_loss_G = 0
            valid_loss_D = 0

            mean_pixel_norm = 0.0
            mean_z_norm = 0
            norm_counter = 1

            for batchID, (data, image_id) in enumerate(tqdm(self.validloader)):
                z = torch.randn(data.shape[0], self.z_dim, 1, 1)
                X = data
                if self.gpu:
                    X, z = data.to(self.device), z.to(self.device)

                z_hat = self.E(X)
                X_hat = self.G(z)

                D_enc = self.D(X,z_hat)
                D_gen = self.D(X_hat,z)

                D_loss = -torch.mean(log(D_enc) + log(1-D_gen))
                G_loss = -torch.mean(log(D_gen) + log(1-D_enc))

                valid_loss_G += G_loss.item()
                valid_loss_D += D_loss.item()

                pixel_norm = X -  self.G(z_hat)
                pixel_norm = pixel_norm.norm().item() / float(self.X_dim)
                mean_pixel_norm += pixel_norm


                z_norm = z - self.E(X_hat)
                z_norm = z_norm.norm().item() / float(self.z_dim)
                mean_z_norm += z_norm

                norm_counter += 1

            print("Eval loss G:", valid_loss_G / norm_counter)
            print("Eval loss D:", valid_loss_D / norm_counter)

            self.eval_hist['D_loss'].append(valid_loss_D / norm_counter)
            self.eval_hist['G_loss'].append(valid_loss_G / norm_counter)

            print("Pixel norm:", mean_pixel_norm / norm_counter)
            self.eval_hist['pixel_norm'].append( mean_pixel_norm / norm_counter )

            with open('pixel_error_BIGAN.txt', 'a') as f:
                f.writelines(str(mean_pixel_norm / norm_counter) + '\n')

            print("z norm:", mean_z_norm / norm_counter)
            self.eval_hist['z_norm'].append( mean_z_norm / norm_counter )

            with open('z_error_BIGAN.txt', 'a') as f:
                f.writelines(str(mean_z_norm / norm_counter) + '\n')

            ### Save X and its reconstruction at the end of each epoch
            samples = X.detach().cpu().numpy()

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

                    # sample = np.clip(sample, 0, 1)
                    sample = sample.reshape(128,128,3)
                    sample = np.rot90(sample, 2)
                    plt.imshow(sample)


            X_hat = self.G(self.E(X))
            samples = X_hat.detach().cpu().numpy()
            for i, sample in enumerate(samples):
                if i<10:
                    ax = plt.subplot(gs[10+i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')

                    # sample = np.clip(sample, 0, 1)
                    sample = sample.reshape(128,128,3)
                    sample = np.rot90(sample, 2)
                    plt.imshow(sample)

            if not os.path.exists(self.result_dir + '/recons/'):
                os.makedirs(self.result_dir + '/recons/')

            filename = "epoch_" + str(epoch)
            plt.savefig(self.result_dir + '/recons/{}.png'.format(filename), bbox_inches='tight')
            plt.close()

    def save_model(self):
        torch.save(self.G.state_dict(), self.save_dir + "/G.pt")
        torch.save(self.E.state_dict(), self.save_dir + "/E.pt")
        torch.save(self.D.state_dict(), self.save_dir + "/D.pt")

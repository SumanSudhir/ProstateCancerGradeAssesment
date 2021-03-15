import os
import torch
import numpy as np
import pandas as pd
import random
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange


class Whole_Slide_Tiles(Dataset):
	def __init__(self,
		dataframe,
		files_path,
		custom_transforms,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.transforms = custom_transforms
		self.dataframe = dataframe
		self.images_id = self.dataframe.image_id.values
		self.files_path = files_path

	def __getitem__(self, idx):
		image_id = self.images_id[idx]
		file_path = os.path.join(self.files_path, image_id + '.h5')

		with h5py.File(file_path,'r') as hdf5_file:
			imgs = np.array(hdf5_file['imgs'])
			coords = np.array(hdf5_file['coords'])

			index = random.randint(0,len(coords)-1)
			img = self.transforms(imgs[index])

		return img, image_id

	def __len__(self):
		return len(self.images_id)



class Whole_Slide_Bag(Dataset):
	def __init__(self,
		dataframe,
		files_path,
		custom_transforms,
		num_patches = 64,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.transforms = custom_transforms
		self.dataframe = dataframe
		self.images_id = self.dataframe.image_id.values
		self.files_path = files_path
		self.num_patches = num_patches


	def __len__(self):
		return len(self.images_id)

	def blue_ratio(self,imgs):
	    allR = []
	    for i in range(len(imgs)):
	        B = imgs[i][:,:,0].mean()
	        G = imgs[i][:,:,1].mean()
	        R = imgs[i][:,:,2].mean()

	        ratio = ((100 * B)/(1+R+G)) * (256/(1+B+R+G))
	        allR.append(ratio)

	    return np.array(allR)

	def expand(self,imgs,coords):

		img = [self.transforms(imgs[i]) for i in range(len(imgs))]
		coord = [coords[i] for i in range(len(imgs))]

		while(len(img) < self.num_patches):
			for i in range(len(imgs)):
				img.append(self.transforms(imgs[i]))
				coord.append(coords[i])

		return img[:self.num_patches], coord[:self.num_patches]

	def __getitem__(self, idx):
		image_id = self.images_id[idx]
		file_path = os.path.join(self.files_path, image_id + '.h5')

		with h5py.File(file_path,'r') as hdf5_file:
			imgs = np.array(hdf5_file['imgs'])
			coords = np.array(hdf5_file['coords'])

			index = np.argsort(-self.blue_ratio(imgs))
			imgs = imgs[index]
			coords = coords[index]

			img, coord = self.expand(imgs,coords)
			img = torch.stack(img)
			# all_imgs = torch.stack(all_imgs)
			label = np.array(self.dataframe[self.dataframe['image_id'] == image_id]['isup_grade']).astype(np.long)
			label_bin = np.zeros(5).astype(np.float32)
			label_bin[:label.squeeze()] = 1

			coord = np.array(coord)

		return img, label_bin, label, coord, image_id


class Whole_Slide_ROI(Dataset):
	def __init__(self,
		dataframe,
		weighted_df,
		files_path,
		custom_transforms,
		num_patches=64
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.transforms = custom_transforms
		self.dataframe = dataframe
		self.images_id = self.dataframe.image_id.values
		self.files_path = files_path
		self.weighted_df = weighted_df
		self.num_patches = num_patches

	def blue_ratio(self,imgs):
	    allR = []
	    for i in range(len(imgs)):
	        B = imgs[i][:,:,0].mean()
	        G = imgs[i][:,:,1].mean()
	        R = imgs[i][:,:,2].mean()

	        ratio = ((100 * B)/(1+R+G)) * (256/(1+B+R+G))
	        allR.append(ratio)

	    return np.array(allR)

	def expand(self,imgs,coords):

		img = [self.transforms(imgs[i]) for i in range(len(imgs))]
		coord = [coords[i] for i in range(len(imgs))]

		while(len(img) < self.num_patches):
			for i in range(len(imgs)):
				img.append(self.transforms(imgs[i]))
				coord.append(coords[i])

		return img[:self.num_patches], coord[:self.num_patches]

	def __getitem__(self, idx):
		image_id = self.images_id[idx]
		file_path = os.path.join(self.files_path, image_id + '.h5')
		weight = self.weighted_df[self.weighted_df["image_id"] == image_id]['weight'].values

		with h5py.File(file_path,'r') as hdf5_file:
			imgs = np.array(hdf5_file['imgs'])
			coords = np.array(hdf5_file['coords'])

			index = np.argsort(-self.blue_ratio(imgs))
			imgs = imgs[index]
			coords = coords[index]

			img, coord = self.expand(imgs,coords)
			img = torch.stack(img)
			w = torch.from_numpy(weight)

			top_p_ids = torch.topk(w, 16)[1]
			top_p = torch.index_select(img, dim=0, index=top_p_ids)
			img_1 = torch.cat([top_p[i] for i in range(0,4)], dim = 2)
			img_2 = torch.cat([top_p[i] for i in range(4,8)], dim = 2)
			img_3 = torch.cat([top_p[i] for i in range(8,12)], dim = 2)
			img_4 = torch.cat([top_p[i] for i in range(12,16)], dim = 2)
			w_img = torch.cat([img_1, img_2, img_3, img_4], dim=1)
			label = np.array(self.dataframe[self.dataframe['image_id'] == image_id]['isup_grade']).astype(np.long)
			label_bin = np.zeros(5).astype(np.float32)
			label_bin[:label.squeeze()] = 1

		return w_img, label_bin, label, image_id

	def __len__(self):
		return len(self.dataframe)

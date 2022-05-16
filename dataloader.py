# from unittest import skip
from torch.utils.data import Dataset, DataLoader
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import cv2

class DataGenerator(Dataset):

	def __init__(self, data_path, split):

		self.files = self.get_files(data_path, split)
		cv2.setNumThreads(0)

	def __len__(self):
		return len(self.files)
		

	def __getitem__(self,idx):

		no_files = len(self.files)

		# while(1):
		# 	try:
				# idx = random.randint(0,no_files)

		split, img_tgt, src_path = self.files[idx]
		img_tgt = cv2.imread(img_tgt)
		img_tgt = cv2.resize(img_tgt,(64, 64))
		img_tgt = (img_tgt/255.)*2 - 1
		img_tgt = img_tgt.transpose((2,0,1))

		src_embed = torch.load(src_path)

		# 	break
		# except:
		# 	continue
		return src_embed, torch.flatten(torch.tensor(img_tgt, dtype=torch.float32))
			
	def get_files(self, data_path, split):

		with open(data_path) as txt:
			lines = txt.readlines()
		data = []
		for l in lines:
			data.append((split, l.split('\t')[0].strip(), l.split('\t')[1].strip()))

		return data
	
def collate_fn_customised(data):

	src_list = []
	tgt_list = []

	#Step 1 : Copying the Data to a List
	for d in data:
		src, tgt = d
		src_list.append(src)
		tgt_list.append(tgt.unsqueeze(0))

	src_tensor = torch.stack(src_list,0).squeeze(1)
	tgt_tensor = torch.stack(tgt_list,0).squeeze(1)

	return src_tensor, tgt_tensor

	
def load_data(data_path, split, batch_size=128, num_workers=1, shuffle=True):

	dataset = DataGenerator(data_path, split)
	data_loader = DataLoader(dataset, collate_fn = collate_fn_customised, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

	return data_loader
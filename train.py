from multiprocessing import reduction
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from generator import *
from discriminator import *
import numpy as np
import random
from torchvision import transforms
from dataloader import *
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F
torch.cuda.empty_cache()



use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

device = torch.device("cuda" if use_cuda else "cpu")

def save_ckp(checkpoint, checkpoint_path):
	torch.save(checkpoint, checkpoint_path)

def load_ckp(checkpoint_path, model, model_opt):
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['state_dict'])
	model_opt.load_state_dict(checkpoint['optimizer'])
	return model, model_opt, checkpoint['epoch']

def train_generator_batch(src_sent, tgt_img, generator, gen_optimizer, gen_loss, discriminator, disc_optimizer, disc_loss):


	generator.train()
	discriminator.eval()

	pred_img = generator.forward(src_sent)
	generator_loss = gen_loss(F.log_softmax(pred_img), F.log_softmax(tgt_img))
	batch_size = pred_img.shape[0]
	discriminator_loss = disc_loss(discriminator.forward(torch.reshape(pred_img,(batch_size, 3, 64, 64)), src_sent), Variable(torch.ones(batch_size, 1)).to(device))
	loss = generator_loss + discriminator_loss	

	return loss, generator, gen_optimizer, generator_loss, discriminator_loss

def train_discriminator_batch(src_sent, tgt_img, generator, discriminator, disc_optimizer, disc_loss):

	discriminator.train()
	generator.eval()

	batch_size = src_sent.shape[0]
	real_img = torch.reshape(tgt_img,(batch_size, 3, 64, 64))
	fake_img = generator.forward(src_sent)
	fake_img = fake_img.detach()
	fake_img = torch.reshape(fake_img,(batch_size, 3, 64, 64))

	batch_size = src_sent.shape[0]
	real_pred = discriminator.forward(real_img, src_sent)
	real_tgt = Variable(torch.ones(batch_size, 1)).to(device)
	real_loss = disc_loss(real_pred, real_tgt)
	
	fake_pred = discriminator.forward(fake_img, src_sent)
	fake_tgt = Variable(torch.zeros(batch_size, 1)).to(device)
	fake_loss = disc_loss(fake_pred, fake_tgt)

	loss = real_loss + fake_loss

	return loss, discriminator, disc_optimizer, real_loss, fake_loss

def train_GAN_epoch(epoch, train_loader, generator, gen_optimizer, gen_loss, discriminator, disc_optimizer, disc_loss):


	progress_bar = tqdm(enumerate(train_loader))
	gen_optimizer.zero_grad()
	disc_optimizer.zero_grad()

	total_gen_loss = 0
	total_disc_loss = 0

	total_real_loss = 0
	total_fake_loss = 0

	for step, (src_sent, tgt_img) in progress_bar:

		src_sent = src_sent.to(device)
		tgt_img = tgt_img.to(device)

		gen_loss_1, generator, gen_optimizer, generator_loss, discriminator_loss = train_generator_batch(src_sent, tgt_img, generator, gen_optimizer, gen_loss, discriminator, disc_optimizer ,disc_loss)
		
		gen_loss_1.backward()
		gen_optimizer.step()
		gen_optimizer.zero_grad()

		disc_loss_1, discriminator, disc_optimizer, real_loss, fake_loss = train_discriminator_batch(src_sent, tgt_img, generator, discriminator, disc_optimizer, disc_loss)

		disc_loss_1.backward()
		disc_optimizer.step()
		disc_optimizer.zero_grad()



		total_gen_loss += generator_loss.item()
		total_disc_loss +=discriminator_loss.item()
		total_real_loss += real_loss.item()
		total_fake_loss += fake_loss.item()

		progress_bar.set_description("Epoch : {}, Gen Loss : {:.4f}, Disc Loss : {:4f}, Real Loss : {:.4f}, Fake Loss : {:.4f}, Iter : {}/{}".format(epoch+1, total_gen_loss/(step+1), total_disc_loss/(step+1), total_real_loss/(step+1), total_fake_loss/(step+1), step+1, len(train_loader)))
		progress_bar.refresh()

	return 	generator, gen_optimizer, discriminator, disc_optimizer




def write_predictions(target_list, predicted_list, epoch):

	output_directory = os.path.join("/scratch/seshadri_c/outputs", "Epoch_" + str(epoch))
	os.makedirs(output_directory, exist_ok=True)

	for i in range(len(target_list)):
		
		img_t, img_p = target_list[i], predicted_list[i]
		
		img_t = (img_t + 1)*255. /2
		img_p = (img_p + 1)*255. /2

		img_t = img_t.astype(np.uint8)
		img_p = img_p.astype(np.uint8)

		plt.figure()
		
		plt.subplot(1,2,1)
		plt.imshow(img_p)
		plt.title("Predicted Image")
		plt.axis("off")

		plt.subplot(1,2,2)
		plt.imshow(img_t)
		plt.title("Target Image")
		plt.axis("off")

		plt.savefig(os.path.join(output_directory, "sample_"+str(i+1)))
		plt.close()

def test_generator_epoch(test_loader, generator, no_samples):

	target_list = []
	predicted_list = []

	generator.eval()

	progress_bar = tqdm(enumerate(test_loader))

	for step, (src_sent, tgt_img) in progress_bar:

		src_sent = src_sent.to(device)
		tgt_img = tgt_img.to(device	)
		pred_img = generator.forward(src_sent)

		batch_size = pred_img.shape[0]

		tgt_img = torch.reshape(tgt_img,(batch_size, 3, 64, 64))
		pred_img = torch.reshape(pred_img,(batch_size, 3, 64, 64))

		for i in range(tgt_img.shape[0]):
			if(len(target_list)<no_samples):
				target_list.append(tgt_img[i].detach().cpu().numpy().transpose((1, 2, 0)))
				predicted_list.append(pred_img[i].detach().cpu().numpy().transpose((1, 2, 0)))

	return target_list, predicted_list

def training_GAN(train_loader, test_loader, valid_loader, generator, gen_optimizer, gen_loss, discriminator, disc_optimizer, disc_loss, resume):


	epoch = 0

	#CHECKPOINT IN SSD SCRATCH
	generator_checkpoint_dir = "/ssd_scratch/cvit/seshadri_c/GAN/Generator/"
	discriminator_checkpoint_dir = "/ssd_scratch/cvit/seshadri_c/GAN/Discriminator/"
	os.makedirs(generator_checkpoint_dir, exist_ok=True)
	os.makedirs(discriminator_checkpoint_dir, exist_ok=True)

	#CHECKPOINT IN SCRATCH
	generator_checkpoint_duplicate_dir = "/scratch/seshadri_c/GAN/Generator/"
	discriminator_checkpoint_duplicate_dir = "/scratch/seshadri_c/GAN/Discriminator/"
	os.makedirs(generator_checkpoint_duplicate_dir, exist_ok=True)
	os.makedirs(discriminator_checkpoint_duplicate_dir, exist_ok=True)

	generator_checkpoint_path = generator_checkpoint_dir + "checkpoint_latest.pt"
	discriminator_checkpoint_path = discriminator_checkpoint_dir + "checkpoint_latest.pt"

	if resume:
		generator, gen_optimizer, epoch = load_ckp(generator_checkpoint_path, generator, gen_optimizer)
		discriminator, disc_optimizer, _ = load_ckp(discriminator_checkpoint_path, discriminator, disc_optimizer)
		resume = False
		print("Resuming Training from Epoch Number : ", epoch)

	discriminator_after_no_steps = 5
	no_samples = 100

	while(1):
	
		generator_checkpoint_duplicate_path = generator_checkpoint_duplicate_dir + "checkpoint_" + str(epoch + 1) + ".pt"
		discriminator_checkpoint_duplicate_path = discriminator_checkpoint_duplicate_dir + "checkpoint_" + str(epoch + 1) + ".pt"

		generator, gen_optimizer, discriminator, disc_optimizer = train_GAN_epoch(epoch, train_loader, generator, gen_optimizer, gen_loss, discriminator, disc_optimizer, disc_loss)

		# Creating the Generator Checkpoint
		generator_checkpoint = {'epoch': epoch, 'state_dict': generator.state_dict(), 'optimizer': gen_optimizer.state_dict()}

		# Saving the Generator Checkpoint
		save_ckp(generator_checkpoint, generator_checkpoint_path)
		print("Generator Weights Saved Successfully")
		save_ckp(generator_checkpoint, generator_checkpoint_duplicate_path)
		print("Generator Weights Duplicate Saved Successfully")

		# Creating the Discriminator Checkpoint
		discriminator_checkpoint = {'epoch': epoch, 'state_dict': discriminator.state_dict(), 'optimizer': disc_optimizer.state_dict()}

		# Saving the Generator Checkpoint
		save_ckp(discriminator_checkpoint, discriminator_checkpoint_path)
		print("Discriminator Weights Saved Successfully")
		save_ckp(discriminator_checkpoint, discriminator_checkpoint_duplicate_path)
		print("Discriminator Weights Saved Successfully")

		target_list, predicted_list = test_generator_epoch(test_loader, generator, no_samples)
		write_predictions(target_list, predicted_list, epoch)

		epoch+=1

def main():

	train_file = "splits/oxford_flower_train.txt"
	test_file = "splits/oxford_flower_test.txt"
	valid_file = "splits/oxford_flower_valid.txt"

	generator = Generator().to(device)
	generator = nn.DataParallel(generator)
	gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
	gen_loss = nn.KLDivLoss(reduction='batchmean',log_target=True).to(device)

	discriminator = Discriminator().to(device)
	discriminator = nn.DataParallel(discriminator)
	disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)
	disc_loss = nn.BCELoss().to(device)


	train_loader = load_data(train_file, "train",batch_size=512,num_workers=10,shuffle=True)
	valid_loader = load_data(valid_file, "valid",batch_size=512,num_workers=10,shuffle=True)
	test_loader = load_data(test_file, "test",batch_size=512,num_workers=10,shuffle=False)

	resume = True

	training_GAN(train_loader, test_loader, valid_loader, generator, gen_optimizer, gen_loss, discriminator, disc_optimizer, disc_loss, resume)

main()
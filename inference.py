# def write_predictions(target_list, predicted_list, epoch):

# 	output_directory = os.path.join("/scratch/seshadri_c/outputs", "Epoch_" + str(epoch))
# 	os.makedirs(output_directory, exist_ok=True)

# 	for i in range(len(target_list)):
		
# 		img_t, img_p = target_list[i], predicted_list[i]
		
# 		img_t = (img_t + 1)*255. /2
# 		img_p = (img_p + 1)*255. /2

# 		img_t = img_t.astype(np.uint8)
# 		img_p = img_p.astype(np.uint8)

# 		plt.figure()
		
# 		plt.subplot(1,2,1)
# 		plt.imshow(img_p)
# 		plt.title("Predicted Image")
# 		plt.axis("off")

# 		plt.subplot(1,2,2)
# 		plt.imshow(img_t)
# 		plt.title("Target Image")
# 		plt.axis("off")

# 		plt.savefig(os.path.join(output_directory, "sample_"+str(i+1)))
# 		plt.close()


# def test_generator_epoch(epoch, test_loader, generator, gen_loss):

# 	no_samples = 200
# 	with torch.no_grad():

# 		generator.eval()

# 		progress_bar = tqdm(enumerate(test_loader))

# 		total_loss = 0

# 		target_list = []
# 		predicted_list = []

# 		for step, (src_sent, tgt_img) in progress_bar:

# 			src_sent = src_sent.to(device)
# 			tgt_img = tgt_img.to(device	)
# 			pred_img = generator.forward(src_sent)

# 			loss = gen_loss(pred_img, tgt_img)

# 			total_loss += loss.item()

# 			progress_bar.set_description("Epoch : {}, Test Loss : {:.4f} Total Iteration : {}/{}".format(epoch+1, total_loss/(step+1), step+1, len(test_loader)))
# 			progress_bar.refresh()

# 			for i in range(tgt_img.shape[0]):
# 				if(len(target_list)<no_samples):
# 					target_list.append(tgt_img[i].cpu().numpy().transpose((1, 2, 0)))
# 					predicted_list.append(pred_img[i].cpu().numpy().transpose((1, 2, 0)))

# 	return target_list, predicted_list
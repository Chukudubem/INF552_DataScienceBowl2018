import pandas as pd
import utils as utils
# from models import unet_model as unet
import importlib
import numpy as np
import models
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
import PIL
import cv2
from skimage.morphology import label # label regions
from skimage.filters import threshold_otsu

def rle_encoding(x):
	'''
	x: numpy array of shape (height, width), 1 - mask, 0 - background
	Returns run length as list
	'''
	dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
	run_lengths = []
	prev = -2
	for b in dots:
		if (b>prev+1): run_lengths.extend((b+1, 0))
		run_lengths[-1] += 1
		prev = b
	return run_lengths

def prob_to_rles(x, cut_off = 0.5):
	lab_img = label(x>cut_off)
	if lab_img.max()<1:
		lab_img[0,0] = 1 # ensure at least one prediction per image
	print(lab_img)
	print(np.unique(lab_img))
	plt.imshow(lab_img)
	for i in range(1, lab_img.max()+1):
		yield rle_encoding(lab_img==i)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Params')
	parser.add_argument('--model_path', nargs='?', type=str, default='models/weights_folder/unet/best_weights.hdf5',
						help='Path to the saved model')
	parser.add_argument('--dataset', nargs='?', type=str, default='nuclei',
						help='Dataset to use [\'nuclei\']')

	args = parser.parse_args()

	# store the parsed args in variables
	model_path = str(Path(args.model_path))
	model_name = model_path.split("/")[-2]
	test_data_path = args.dataset

	if test_data_path == "nuclei":
		test_data_path = str(Path("../data/stage1_test"))
		# save test_concatenated_labels
		save_path = str(Path("../data/stage1_test_concatenated_labels"))

	if test_data_path == "nuclei_stage_2":
		test_data_path = str(Path("../data/stage2_test"))
		# save test_concatenated_labels
		save_path = str(Path("../data/stage2_test_concatenated_labels"))

	# Load the test data
	X_test, image_names, og_size = utils.load_test_data(test_data_path)

	# get the model using the string name
	model = importlib.import_module("models.unet_model").get_unet_256(num_classes=1)
	model.summary()

	output_to_file = []

	for image, name, op_size in zip(X_test, image_names, og_size):
		image = np.expand_dims(image, axis=0)
		predicted_mask = model.predict(image)
		predicted_mask = predicted_mask.squeeze(axis = 0)
		# print(predicted_mask.shape)
		# print(name)
		# print(predicted_mask)
		# plt.imshow(predicted_mask.squeeze())
		predicted_mask = 255 - (predicted_mask.squeeze()*255).astype(np.uint8)
		# thresh = threshold_otsu(predicted_mask)
		# print(thresh)
		# print(np.unique(predicted_mask))

		ret, predicted_mask = cv2.threshold(predicted_mask, np.max(predicted_mask)-50,\
											255, cv2.THRESH_BINARY)

		# predicted_mask = cv2.adaptiveThreshold(predicted_mask, np.max(predicted_mask),\
		# 									cv2.ADAPTIVE_THRESH_MEAN_C,\
		# 									cv2.THRESH_BINARY, 21, 0)

		# print(op_size)
		# print(np.unique(predicted_mask))
		predicted_mask = cv2.resize(predicted_mask, (op_size))

		# experiment ###################################################
		# gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		# predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2GRAY)
		# predicted_mask = gray_image
		# predicted_mask = cv2.resize(predicted_mask, (op_size))
		# print(predicted_mask.shape)
		# plt.imshow(predicted_mask)
		# ret, predicted_mask = cv2.threshold(predicted_mask, np.max(predicted_mask)-50,\
		# 									255, cv2.THRESH_BINARY)
		# plt.imshow(predicted_mask)
		##########################################################


		predicted_mask_image = PIL.Image.fromarray(predicted_mask)
		# predicted_mask_image.show()
		rles = list(prob_to_rles(predicted_mask))

		count = 0
		for rle in rles:
			output_to_file.append({"ImageId":name.split('.')[0],\
								   "EncodedPixels":" ".join(np.array(rle).astype(str))})
			print(count)
			count += 1
		# break

	output_pred_df = pd.DataFrame(output_to_file)
	cols = ['ImageId', 'EncodedPixels']
	output_pred_df = output_pred_df[cols]
	print(output_pred_df)
	output_pred_df.to_csv('../data/stage2_submission/submission.csv', index=None)
	# plt.show()

from skimage.morphology import label # label regions
import cv2
import numpy as np
from models.data_augmentations import randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip
from models import unet_model as unet
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import glob
import PIL
from pathlib import Path
import joblib
import h5py as h5py
from sklearn.model_selection import train_test_split

def convert_path(path):
	"""
	# convert paths to the os required strings using Pathlib's Path module
	:param path:
	:return:
	String(path)
	"""
	return Path(path)

# Constants later shift this to __init__.py
DATA_PATH = convert_path("../data/")
STAGE1_TRAIN_DATA = convert_path("{DATA_PATH}/stage1_train".format(DATA_PATH=DATA_PATH))
STAGE1_TRAIN_LABELS = convert_path("{DATA_PATH}/stage1_train_labels".format(DATA_PATH=DATA_PATH))
STAGE1_TEST_DATA = convert_path("{DATA_PATH}/stage1_test".format(DATA_PATH=DATA_PATH))
STAGE1_TEST_LABELS = ""
STAGE2_TEST_DATA = convert_path("{DATA_PATH}/stage2_test".format(DATA_PATH=DATA_PATH))


def load_image_labels(folder, border_sz=1):
	"""
	Load
	:param folder:
	:param border_sz:
	:return:
	"""
	image = glob.glob(folder + '/images/*')[0]
	image = cv2.imread(image)[:, :, ::-1]
	masks = glob.glob(folder + '/masks/*')
	all_masks = []
	for i, mask in enumerate(masks):
		mask_img = np.sum(cv2.imread(mask), axis=-1)
		mask_img = cv2.erode(mask_img.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
		all_masks.append((mask_img.astype(np.int16) * (i + 1)))
	if len(masks)==0:
		return image
	return image, np.sum(all_masks, axis=0, dtype=np.int16)


def fix_mask(mask):
	mask[mask<80] = 0.0
	mask[mask>=80] = 255.0

def train_process(sample):
	size = (256, 256)
	img, mask = sample
	img = img[:, :, :3]
	fix_mask(mask)
	img = cv2.resize(img, size)
	mask = cv2.resize(mask, size)
	img = randomHueSaturationValue(img,
								   hue_shift_limit=(-50, 50),
								   sat_shift_limit=(0, 0),
								   val_shift_limit=(-15, 15))
	img, mask = randomShiftScaleRotate(img, mask,
									   shift_limit=(-0.062, 0.062),
									   scale_limit=(-0.1, 0.1),
									   rotate_limit=(-20, 20))
	img, mask = randomHorizontalFlip(img, mask)
	fix_mask(mask)
	img = img / 255.
	mask = mask / 255.

	mask = np.expand_dims(mask, axis=2)
	return (img, mask)

def load_test_data(path):
	"""
	:param path: path of the test data
	:return:
	x_pixels => pixels of images
	image_name => name of images
	"""
	size = (256, 256)
	x_pixels = []
	image_name = []
	og_size = []
	for path in glob.glob("{}/*/images/*".format(path)):
		# print(path)
		image_name.append(path.split('/')[-1])
		image = cv2.imread(path)
		og_size.append((image.shape[0], image.shape[1]))
		image = cv2.resize(image, size)
		x_pixels.append(image)
	#
	return x_pixels, image_name, og_size

def load_train_data():
	"""
	Loads the train data
	:return:
	x_pixels, y_pixels => images as an array of pixels (list)
	"""
	# x = []
	# x_pixels = []
	# y = []
	# y_pixels = []
	x_pixels_mod = []
	y_pixels_mod = []

	for path in glob.glob("{STAGE1_TRAIN_DATA}/*/".format(STAGE1_TRAIN_DATA=STAGE1_TRAIN_DATA)):
		image, mask = load_image_labels(path)
		image, mask = train_process((image, mask))
		x_pixels_mod.append(image)
		y_pixels_mod.append(mask)

	return x_pixels_mod, y_pixels_mod

def data_generator(X, y, batch_size=1, shuffle=False):
	"""
	generates data with a default batch size of 1
	:param X:
	:param y:
	:param batch_size:
	:param shuffle:
	:return:
	"""
	obj = list(zip(X, y))
	if shuffle == True:
		np.random.shuffle(obj)
	counter = 0
	if counter + batch_size >= len(obj):
		counter = len(obj) - 1 - batch_size
	img, mask = zip(*obj[counter:counter + batch_size])
	counter = (counter + batch_size) % len(obj)
	while True:
		yield np.array(img), np.array(mask)

def generate_prediction_for_one(b, model):
	"""
	Generates prediction for one image and displays both the image
	and the generated mask for the users perusal
	:param b: for train and validation data pass tuple (image, mask), for test image without masks pass tuple (image, mask)
	:param model:
	:return:
	"""
	image_batch, mask_batch = next(b)
	predicted_mask_batch = model.predict(image_batch)
	image = image_batch[0]
	print(image.shape, predicted_mask_batch.shape)
	return image, predicted_mask_batch

def concatenate_masks():
	"""
	concatenate all the masks for one image into one PIL.Imagefromarray
	:return:
	"""
	pass

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
        print(run_lengths)
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cut_off = 0.5):
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)


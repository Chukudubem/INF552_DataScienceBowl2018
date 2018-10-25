import pandas as pd
import utils as utils
from models import unet_model as unet
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

unet_callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='models/weights_folder/unet/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True)]

if __name__ == "__main__":

	# Load the train data
	X, y = utils.load_train_data()
	print(X, y)
	model = unet.get_unet_256(num_classes=1)
	model.summary()

	X_train, X_validation, y_train, y_validation = \
        train_test_split(X,\
                         y,\
                         train_size = 0.6)

	epochs = 1

	train_generator = utils.data_generator(X_train, y_train)

	validation_generator = utils.data_generator(X_validation, y_validation)

	model.fit_generator(generator=train_generator, \
						steps_per_epoch=1000, \
						epochs=epochs, \
						callbacks=unet_callbacks, \
						validation_data=validation_generator, \
						validation_steps=len(X_validation)
						)

	image, predict_mask = utils.generate_prediction_for_one(validation_generator, model)
	plt.imshow(image)
	plt.imshow(predict_mask.squeeze())
	plt.show()


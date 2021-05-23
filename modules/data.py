#!/usr/bin/env python
""" HIAS AI Model Data Class.

Provides the HIAS AI Model with the required required data
processing functionality.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contributors:
- Adam Milton-Barker

"""

import os
import pathlib

import pandas as pd
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model

from modules.AbstractData import AbstractData

class data(AbstractData):
	""" HIAS AI Model Data Class.

	Provides the HIAS AI Model with the required required data
	processing functionality.
	"""

	def pre_process(self):
		""" Processes the images. """

		#Load VGG-16 model
		model = VGG19(weights='imagenet', include_top= True )
		layer_name = 'fc2'
		intermediate_layer_model = keras.Model(
			inputs=model.input,outputs=model.get_layer(layer_name).output)
		intermediate_layer_model.summary()

		#Load the data directory  where the images are stored
		data_dir = pathlib.Path(
			self.helpers.confs["data"]["train_dir"])
		data = list(data_dir.glob(
			'*' + self.helpers.confs["data"]["file_type"]))
		data.sort()

		images = []
		batch = []
		labels = []

		neg_count = 0
		pos_count = 0

		j = 0

		for i, rimage in enumerate(data, 1):
			fpath = str(rimage)
			fname = os.path.basename(rimage)
			label = "Non-Covid" if "Non" in fname else "Covid"
			j = 0 if "Non" in fname else 1

			self.helpers.logger.info("Processing: " + fname)
			self.helpers.logger.info("Label: " + label)
			self.helpers.logger.info("Class: " + str(j))

			img = image.load_img(fpath, target_size=(self.dim, self.dim))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			# Extract features using the VGG-16 structure
			features = intermediate_layer_model.predict(x)
			# Append features and labels
			batch.append(features[0])
			labels.append(fpath + ';' + str(j))

			if "Non" in fname:
				neg_count += 1
			else:
				pos_count += 1

		self.np_batch = np.array(batch)
		self.np_labels = np.array(labels)

		self.helpers.logger.info("Negative data: " + str(neg_count))
		self.helpers.logger.info("Positive data: " + str(pos_count))
		self.helpers.logger.info("Batch: " + str(self.np_batch.shape))
		self.helpers.logger.info("Labels: " + str(self.np_labels.shape))

		self.get_split()

		#Convert data to Pandas in order to save as .csv
		data_df_X_train = pd.DataFrame(self.X_train)
		data_df_y_train = pd.DataFrame(self.y_train)
		data_df_X_test = pd.DataFrame(self.X_test)
		data_df_y_test = pd.DataFrame(self.y_test)

		# Save file as .csv
		data_df_X_train.to_csv(
			self.helpers.confs["model"]["features"] + '/data_df_X_train_lite.csv', header=False, index=False)
		data_df_y_train.to_csv(
			self.helpers.confs["model"]["features"] + '/data_df_y_train_lite.csv', header=False, index=False)
		data_df_X_test.to_csv(
			self.helpers.confs["model"]["features"] + '/data_df_X_test_lite.csv', header=False, index=False)
		data_df_y_test.to_csv(
			self.helpers.confs["model"]["features"] + '/data_df_y_test_lite.csv', header=False, index=False)

	def convert_data(self):
		""" Converts the training data to a numpy array. """
		pass

	def encode_labels(self):
		""" One Hot Encodes the labels. """
		pass

	def shuffle(self):
		""" Shuffles the data and labels. """
		pass

	def get_split(self):
		""" Splits the data and labels creating training and validation datasets. """

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			self.np_batch, self.np_labels, test_size=0.2, random_state=0)

		self.helpers.logger.info("Training data: " + str(self.X_train.shape))
		self.helpers.logger.info("Training labels: " + str(self.y_train.shape))
		self.helpers.logger.info("Validation data: " + str(self.X_test.shape))
		self.helpers.logger.info("Validation labels: " + str(self.y_test.shape))

	def resize(self, path, dim):
		""" Resizes an image to the provided dimensions (dim). """

		processed = img.resize((dim, dim))

		return processed

	def reshape(self, img):
		""" Classifies an image sent via HTTP. """
		pass

	def ext_feature(self, img):
		"""  Extract feature from image """

		x = keras.preprocessing.image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		features = self.tf_intermediate_layer_model.predict(x)
		test = []
		test.append(features[0])
		np_feature = np.array(test)

		return np_feature





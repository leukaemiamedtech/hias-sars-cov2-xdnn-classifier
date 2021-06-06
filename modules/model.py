#!/usr/bin/env python
""" Class representing a HIAS AI Model.

Represents a HIAS AI Model. HIAS AI Models are used by AI Agents to process
incoming data.

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
- Nitin Mane

"""

import cv2
import json
import os
import random
import requests
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scikitplot as skplt

from PIL import Image
from numpy.random import seed
from numpy import genfromtxt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow import keras

from modules.xDNN_class import *
from modules.xDNN_class import xDNN

from modules.AbstractModel import AbstractModel


class model(AbstractModel):
	""" Class representing a HIAS AI Model.

	This object represents a HIAS AI Model.HIAS AI Models are used by AI Agents
	to process incoming data.
	"""

	def prepare_data(self):
		""" Creates/sorts dataset. """

		self.data.remove_testing()
		self.data.pre_process()

		self.helpers.logger.info("Data preperation complete.")

	def prepare_network(self):
		""" Prepares the network. """

		self.model = VGG19(weights='imagenet', include_top=True)
		layer_name = 'fc2'

		self.load()
		self.tf_intermediate_layer_model = keras.Model(
			inputs=self.model.input, outputs=self.model.get_layer(layer_name).output, name="SarsCov2xDNN")

		self.load()
		self.tf_intermediate_layer_model.summary()

		self.helpers.logger.info("VGG19 Model loaded ")

	def train(self):
		""" Trains the model

		Trains the neural network.
		"""

		# Load the files, including features, images and labels.

		X_train_file_path = r'./' + self.confs["model"]["features"] + '/data_df_X_train_lite.csv'
		y_train_file_path = r'./' + self.confs["model"]["features"] + '/data_df_y_train_lite.csv'
		X_test_file_path = r'./' + self.confs["model"]["features"] + '/data_df_X_test_lite.csv'
		y_test_file_path = r'./' + self.confs["model"]["features"] + '/data_df_y_test_lite.csv'

		X_train = genfromtxt(X_train_file_path, delimiter=',')
		y_train = pd.read_csv(y_train_file_path, delimiter=';',header=None)
		X_test = genfromtxt(X_test_file_path, delimiter=',')
		self.X_test = X_test
		y_test = pd.read_csv(y_test_file_path, delimiter=';',header=None)

		# Print the shape of the data

		self.helpers.logger.info(
			"###################### Data Loaded ######################")
		self.helpers.logger.info("X train: " + str(X_train.shape))
		self.helpers.logger.info("Y train: " + str(y_train.shape))
		self.helpers.logger.info("X test: " + str(X_test.shape))
		self.helpers.logger.info("Y test: " + str(y_test.shape))

		pd_y_train_labels = y_train[1]
		pd_y_train_images = y_train[0]

		pd_y_test_labels = y_test[1]
		pd_y_test_images = y_test[0]

		# Convert Pandas to Numpy
		y_train_labels = pd_y_train_labels.to_numpy()
		y_train_images = pd_y_train_images.to_numpy()

		y_test_labels = pd_y_test_labels.to_numpy()
		self.y_test_labels = y_test_labels
		y_test_images = pd_y_test_images.to_numpy()

		# Model Learning
		Input1 = {}

		Input1['Images'] = y_train_images
		Input1['Features'] = X_train
		Input1['Labels'] = y_train_labels

		Mode1 = 'Learning'

		start = time.time()
		Output1 = xDNN(Input1,Mode1)

		end = time.time()

		self.helpers.logger.info("Time: " + str(round(end - start, 2)) + "seconds")
		self.helpers.logger.info(
			"###################### Model Loaded ####################")

		# Load the files, including features, images and labels for the validation mode

		Input2 = {}
		Input2['xDNNParms'] = Output1['xDNNParms']
		Input2['Images'] = y_test_images
		Input2['Features'] = X_test
		Input2['Labels'] = y_test_labels

		startValidation = time.time()
		Mode2 = 'Validation'
		Output2 = xDNN(Input2,Mode2)
		endValidation = time.time()

		self.Output2 = Output2

		self.helpers.logger.info(
			"###################### Results ####################")

		# Elapsed Time
		self.helpers.logger.info(
			"Time: " + str(round(endValidation - startValidation, 2)) + "seconds")

		# accuracy: (tp + tn) / (p + n)
		self.accuracy = accuracy_score(y_test_labels, Output2['EstLabs'])
		self.helpers.logger.info('Accuracy: ' + str(self.accuracy))

		# precision tp / (tp + fp)
		self.precision = precision_score(
			y_test_labels, Output2['EstLabs'], average='micro')
		self.helpers.logger.info('Precision: ' + str(self.precision))

		# recall: tp / (tp + fn)
		self.recall = recall_score(
			y_test_labels, Output2['EstLabs'], average='micro')
		self.helpers.logger.info('Recall: ' + str(self.recall))

		# f1: 2 tp / (2 tp + fp + fn)
		self.f1 = f1_score(y_test_labels,
							Output2['EstLabs'],  average='micro')
		self.helpers.logger.info('F1 score: ' + str(self.f1))

		# kappa
		self.kappa = cohen_kappa_score(y_test_labels, Output2['EstLabs'])
		self.helpers.logger.info('Cohens kappa: ' + str(self.kappa))

		# Recall, Precision and F1

		self.visualize_metrics()
		self.confusion_matrix()
		self.figures_of_merit()

	def freeze_model(self):
		""" Freezes the model """
		pass

	def save_model_as_json(self):
		""" Saves the model as JSON """
		pass

		self.helpers.logger.info("Model JSON saved " + self.model_json)

	def save_weights(self):
		""" Saves the model weights """
		pass

	def visualize_metrics(self):
		""" Visualize the metrics. """

		# Pie chart, where the slices will be ordered and plotted counter-clockwise:
		labels = ['COVID', 'Non-COVID']
		inaccurate = (1 - self.accuracy)
		sizes = [self.accuracy, inaccurate]
		explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
				shadow=True, startangle=90)
		ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
		ax1.set_title('Accuracy Chart');
		plt.savefig('model/plots/accuracy_chart.png')

		# Recall, Precision and F1
		fig2, ax2 = plt.subplots()
		ax2 = fig2.add_axes([0,0,1,1])
		name = ['precision', 'Recall', 'F1 score']
		values = [self.precision, self.recall, self.f1]
		ax2.bar(name,values)
		ax2.set_title('Precision - Recall - F1')
		plt.savefig('model/plots/recall.png', bbox_inches='tight')

	def confusion_matrix(self):
		""" Prints/displays the confusion matrix. """

		# confusion matrix
		self.matrix = confusion_matrix(self.y_test_labels, self.Output2['EstLabs'])
		self.helpers.logger.info("Confusion Matrix: " + str(self.matrix))
		print("")

		per_matrix = self.matrix.astype(
			'float') / self.matrix.sum(axis=1)[:, np.newaxis]
		self.helpers.logger.info("Normalized Confusion Matrix: " + str(per_matrix))
		print("")

		fig3 = plt.figure()
		con_mat = skplt.metrics.plot_confusion_matrix(
			self.y_test_labels,
			self.Output2['EstLabs'],
			figsize=(8, 8))
		plt.savefig('model/plots/confusion_matrix.png', bbox_inches='tight')

	def figures_of_merit(self):
		""" Calculates/prints the figures of merit.

		https://homes.di.unimi.it/scotti/all/
		"""

		test_len = len(self.X_test)

		TP = self.matrix[1][1]
		TN = self.matrix[0][0]
		FP = self.matrix[0][1]
		FN = self.matrix[1][0]

		TPP = (TP * 100)/test_len
		FPP = (FP * 100)/test_len
		FNP = (FN * 100)/test_len
		TNP = (TN * 100)/test_len

		specificity = TN/(TN+FP)

		misc = FP + FN
		miscp = (misc * 100)/test_len

		self.helpers.logger.info(
			"True Positives: " + str(TP) + "(" + str(TPP) + "%)")
		self.helpers.logger.info(
			"False Positives: " + str(FP) + "(" + str(FPP) + "%)")
		self.helpers.logger.info(
			"True Negatives: " + str(TN) + "(" + str(TNP) + "%)")
		self.helpers.logger.info(
			"False Negatives: " + str(FN) + "(" + str(FNP) + "%)")

		self.helpers.logger.info("Specificity: " + str(specificity))
		self.helpers.logger.info("Misclassification: " +
						str(misc) + "(" + str(miscp) + "%)")

	def load(self):
		""" Load the files, including features, images and labels. """

		X_train_file_path = self.helpers.confs["model"]["features"] + \
			'/data_df_X_train_lite.csv'
		y_train_file_path = self.helpers.confs["model"]["features"] + \
			'/data_df_y_train_lite.csv'

		X_train = genfromtxt(X_train_file_path, delimiter=',')
		y_train = pd.read_csv(y_train_file_path, delimiter=';',header=None)

		self.helpers.logger.info("###################### Data Loaded ######################")
		self.helpers.logger.info("X train: " + str(X_train.shape))
		self.helpers.logger.info("Y train: " + str(y_train.shape))

		pd_y_train_labels = y_train[1]
		pd_y_train_images = y_train[0]

		# Convert Pandas to Numpy
		y_train_labels = pd_y_train_labels.to_numpy()
		y_train_images = pd_y_train_images.to_numpy()

		# Model Learning
		Input1 = {}

		Input1['Images'] = y_train_images
		Input1['Features'] = X_train
		Input1['Labels'] = y_train_labels

		Mode1 = 'Learning'

		self.Output1 = xDNN(Input1, Mode1)

		self.helpers.logger.info("Loaded xDNN Model.")

	def evaluate(self):
		""" Evaluates the model """
		pass

	def predictions(self):
		""" Gets a prediction for an image. """
		pass

	def predict(self, out_fe):
		""" Gets a prediction for output features. """

		Input3 = {}
		Input3['xDNNParms'] = self.Output1['xDNNParms']
		Input3['Features'] = out_fe
		Mode3 = 'classify'

		Output3 = xDNN(Input3,Mode3)
		out1 = Output3['Scores'][0][0]
		out2 = Output3['Scores'][0][1]

		if out1 < out2:
			prediction = 1
		else:
			prediction = 0

		return prediction

	def reshape(self, img):
		""" Reshapes an image. """
		pass

	def test(self):
		""" Test mode

		Loops through the test directory and classifies the images.
		"""

		totaltime = 0
		files = 0

		tp = 0
		fp = 0
		tn = 0
		fn = 0
		prediction = 0

		for testFile in os.listdir(self.testing_dir):
			if os.path.splitext(testFile)[1] in self.valid:

				fileName = self.testing_dir + "/" + testFile

				img = tf.keras.preprocessing.image.load_img(
					fileName, target_size=(224, 224), color_mode='rgb')
				out_fe = self.ext_feature(img)
				start = time.time()
				prediction = self.predict(out_fe)
				end = time.time()
				benchmark = end - start
				totaltime += benchmark

				msg = ""
				status = ""
				outcome = ""

				if prediction == 1 and "Non-Covid" in testFile:
					fp += 1
					status = "incorrectly"
					outcome = "(False Positive)"

				elif prediction == 0 and "Non-Covid" in testFile:
					tn += 1
					status = "correctly"
					outcome = "(True Negative)"

				elif prediction == 1 and "Covid" in testFile:
					tp += 1
					status = "correctly"
					outcome = "(True Positive)"

				elif prediction == 0 and "Covid" in testFile:
					fn += 1
					status = "incorrectly"
					outcome = "(False Negative)"

				files += 1
				self.helpers.logger.info("SARS-CoV-2 xDNN Classifier " + status +
								" detected " + outcome + " in " + str(benchmark) + " seconds.")

		self.helpers.logger.info("Images Classified: " + str(files))
		self.helpers.logger.info("True Positives: " + str(tp))
		self.helpers.logger.info("False Positives: " + str(fp))
		self.helpers.logger.info("True Negatives: " + str(tn))
		self.helpers.logger.info("False Negatives: " + str(fn))
		self.helpers.logger.info("Total Time Taken: " + str(totaltime))

	def test_http(self):
		""" HTTP test mode

		Loops through the test directory and classifies the images
		by sending data to the classifier using HTTP requests.
		"""

		totaltime = 0
		files = 0

		tp = 0
		fp = 0
		tn = 0
		fn = 0

		self.addr = "http://" + self.helpers.credentials["server"]["ip"] + \
			':'+str(self.helpers.credentials["server"]["port"]) + '/Inference'
		self.headers = {'content-type': 'image/jpeg'}

		for testFile in os.listdir(self.testing_dir):
			if os.path.splitext(testFile)[1] in self.valid:

				start = time.time()
				prediction = self.http_request(self.testing_dir + "/" + testFile)
				end = time.time()
				benchmark = end - start
				totaltime += benchmark

				msg = ""
				status = ""
				outcome = ""

				print()

				if prediction["Diagnosis"] == "Positive" and "Non-Covid" in testFile:
					fp += 1
					status = "incorrectly"
					outcome = "(False Positive)"

				elif prediction["Diagnosis"] == "Negative" and "Non-Covid" in testFile:
					tn += 1
					status = "correctly"
					outcome = "(True Negative)"

				elif prediction["Diagnosis"] == "Positive" and "Covid" in testFile:
					tp += 1
					status = "correctly"
					outcome = "(True Positive)"

				elif prediction["Diagnosis"] == "Negative" and "Covid" in testFile:
					fn += 1
					status = "incorrectly"
					outcome = "(False Negative)"

				files += 1
				self.helpers.logger.info("SARS-CoV-2 " + status +
								" detected " + outcome + " in " + str(benchmark) + " seconds.")

		self.helpers.logger.info("Images Classified: " + str(files))
		self.helpers.logger.info("True Positives: " + str(tp))
		self.helpers.logger.info("False Positives: " + str(fp))
		self.helpers.logger.info("True Negatives: " + str(tn))
		self.helpers.logger.info("False Negatives: " + str(fn))
		self.helpers.logger.info("Total Time Taken: " + str(totaltime))

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

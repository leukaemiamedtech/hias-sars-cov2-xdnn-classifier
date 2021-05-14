#!/usr/bin/env python
###################################################################################
##
## Project:    COVID -19 xDNN Classifier 2020
## Version:    1.0.0
## Module:     Server
## Desription: The COVID -19 xDNN Classifier 2020 server.
## License:    MIT
## Copyright:  2021, Asociacion De Investigacion En Inteligencia Artificial Para
##             La Leucemia Peter Moss.
## Author:     Nitin Mane
## Maintainer: Nitin Mane
##
## Modified:   2021-2-19
##
###################################################################################
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files(the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
###################################################################################

import cv2
import json
import jsonpickle
import os
import requests
import time

import numpy as np
import tensorflow as tf

from modules.AbstractServer import AbstractServer

from flask import Flask, request, Response
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


class server(AbstractServer):
	""" COVID 19 xDNN Classifier 2020 Server.

	This object represents the COVID 19 xDNN Classifier 2020 Server.
	"""

	def predict(self, req):
		""" Classifies an image sent via HTTP. """

		if len(req.files) != 0:
			img = Image.open(req.files['file'].stream).convert('RGB')
		else:
			img = Image.open(BytesIO(req.data)).convert('RGB')

		img = img.resize((224, 224), Image.ANTIALIAS)
		np_img = tf.keras.preprocessing.image.img_to_array(img)
		np_img.transpose(1, 2, 0)
		#img = keras.preprocessing.image.img_to_array(img)
		#img = np.array([img])  # Convert single image to a batch.
		img = np.expand_dims(np_img, axis=0)
		img = preprocess_input(img)
		#prediction = self.predict(img)

		#img = img.resize((224, 224), Image.ANTIALIAS)
		#img = image.img_to_array(img)
		#img = np.expand_dims(img, axis=0)
		#img = preprocess_input(img)
		#img = img.reshape((1,224,224,3))

		return self.model.predict(img)

	def request(self, img_path):
		""" Sends image to the inference API endpoint. """

		self.helpers.logger.info("Sending request for: " + img_path)

		_, img_encoded = cv2.imencode('.png', cv2.imread(img_path))
		response = requests.post(
			self.addr, data=img_encoded.tostring(), headers=self.headers)
		response = json.loads(response.text)

		return response

	def start(self):
		""" Starts the server. """

		app = Flask(self.helpers.credentials["iotJumpWay"]["name"])

		@app.route('/Inference', methods=['POST'])
		def Inference():
			""" Responds to HTTP POST requests. """

			self.mqtt.publish("States", {
				"Type": "Prediction",
				"Name": self.helpers.credentials["iotJumpWay"]["name"],
				"State": "Processing",
				"Message": "Processing data"
			})

			message = ""
			prediction = self.predict(request)
			print(prediction)

			if prediction == 1:
				message = "Acute Lymphoblastic Leukemia detected!"
				diagnosis = "Positive"
			elif prediction == 0:
				message = "Acute Lymphoblastic Leukemia not detected!"
				diagnosis = "Negative"

			self.mqtt.publish("States", {
				"Type": "Prediction",
				"Name": self.helpers.credentials["iotJumpWay"]["name"],
				"State": diagnosis,
				"Message": message
			})

			resp = jsonpickle.encode({
				'Response': 'OK',
				'Message': message,
				'Diagnosis': diagnosis
			})

			return Response(response=resp, status=200, mimetype="application/json")

		app.run(host=self.helpers.credentials["server"]["ip"],
				port=self.helpers.credentials["server"]["port"])

	def test(self):
		""" Tests the trained model via HTTP. """

		totaltime = 0
		files = 0

		tp = 0
		fp = 0
		tn = 0
		fn = 0

		self.addr = "http://" + self.helpers.credentials["server"]["ip"] + \
			':'+str(self.helpers.credentials["server"]["port"]) + '/Inference'
		self.headers = {'content-type': 'image/jpeg'}

		for testFile in os.listdir(self.model.testing_dir):
			if os.path.splitext(testFile)[1] in self.model.valid:

				start = time.time()
				prediction = self.request(self.model.testing_dir + "/" + testFile)
				print(prediction)
				end = time.time()
				benchmark = end - start
				totaltime += benchmark

				msg = ""
				status = ""
				outcome = ""

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
				self.helpers.logger.info("COVID-19 xDNN Classifier " + status +
									" detected " + outcome + " in " + str(benchmark) + " seconds.")

		self.helpers.logger.info("Images Classified: " + str(files))
		self.helpers.logger.info("True Positives: " + str(tp))
		self.helpers.logger.info("False Positives: " + str(fp))
		self.helpers.logger.info("True Negatives: " + str(tn))
		self.helpers.logger.info("False Negatives: " + str(fn))
		self.helpers.logger.info("Total Time Taken: " + str(totaltime))

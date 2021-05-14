#!/usr/bin/env python
###################################################################################
##
## Project:    HIAS Hospital Intelligent Automation Server
## Version:    3.0.0
## Module:     HIAS AI Server Abstract Class
## Desription: HIAS AI Agent Server class.
## License:    MIT
## Copyright:  2021, Asociacion De Investigacion En Inteligencia Artificial Para
##             La Leucemia Peter Moss.
## Author:     Adam Milton-Barker
## Maintainer: Adam Milton-Barker
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

from abc import ABC, abstractmethod

class AbstractServer(ABC):
	""" Abstract class representing a HIAS AI Agent Server.

	This object represents a HIAS AI Agent Server.
	"""

	def __init__(self, helpers, model, mqtt):
		"Initializes the AbstractModel object."
		super().__init__()

		self.helpers = helpers
		self.confs = self.helpers.confs

		self.model = model

		if mqtt is not None:
			self.mqtt = mqtt

		self.helpers.logger.info("Agent initialization complete.")


	@abstractmethod
	def predict(self, req):
		""" Classifies an image sent via HTTP. """
		pass

	@abstractmethod
	def request(self, img_path):
		""" Sends image to the inference API endpoint. """
		pass

	@abstractmethod
	def start(self, img_path):
		""" Sends image to the inference API endpoint. """
		pass

	@abstractmethod
	def test(self, img_path):
		""" Sends image to the inference API endpoint. """
		pass

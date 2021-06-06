#!/bin/bash

FMSG="- SARS-CoV-2 xDNN Classifier installation terminated"

read -p "? This script will install the SARS-CoV-2 xDNN Classifier required Python libraries on your device. Are you ready (y/n)? " cmsg

if [ "$cmsg" = "Y" -o "$cmsg" = "y" ]; then

	echo "- Installing required Python libraries"
	pip3 install --user urllib3
	pip3 install --user pandas
	pip3 install --user pathlib
	pip3 install --user numpy
	pip3 install --user matplotlib
	pip3 install --user scikit-plot
	pip3 install --user scikit-learn
	pip3 install --user Flask
	pip3 install --user Werkzeug
	pip3 install --user WSGIserver
	pip3 install --user gevent

else
	echo $FMSG;
	exit
fi
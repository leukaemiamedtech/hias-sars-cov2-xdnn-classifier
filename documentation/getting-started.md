# Asociación de Investigacion en Inteligencia Artificial Para la Leucemia Peter Moss
## SARS-CoV-2 xDNN Classifier
### Getting Started

![SARS-CoV-2 xDNN Classifier](../assets/images/covid-xdnn-hias.jpg)

&nbsp;

# Table Of Contents

- [Introduction](#introduction)
    - [Network Architecture](#network-architecture)
- [Installation](#installation)
- [Data](#data)
	- [Application Testing Data](#application-testing-data)
- [Configuration](#configuration)
- [Training](#training)
    - [Metrics](#metrics)
	- [Start Training](#start-training)
	- [Training Data](#training-data)
	- [Model Summary](#model-summary)
	- [Training Results](#training-results)
	- [Metrics Overview](#metrics-overview)
	- [ALL-IDB Required Metrics](#all-idb-required-metrics)
- [Testing](#testing)
- [OpenVINO Intermediate Representation](#openvino-intermediate-representation)
- [HIAS UI](#hias-ui)
    - [NGINX](#nginx)
    - [Inference](#inference)
      - [Verification](#verification)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;

# Introduction
This guide will guide you through the installation process for the SARS-CoV-2 xDNN Classifier.

## Network Architecture
You will build an xPlainable Deep Neural Network based on the proposed architecture in [SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification](https://www.medrxiv.org/content/10.1101/2020.04.24.20078584v3).

&nbsp;

# Installation
First you need to install the required software for training the model. Below are the available installation guides:

- [Ubuntu installation guide](installation/ubuntu.md) (Training).

&nbsp;

# Data
You need to download a copy of the Now Download the [SARS-CoV-2 CT Scan Dataset](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset) to use the SARS-CoV-2 xDNN Classifier.

Once you have your data you need to add it to the project filesystem. You will notice the data folder in the Model directory, **model/data**, inside you have **train** & **test**. Add all of the images from the **COVID** & **non-COVID** directories to the **model/data/train** directory.

## Application testing data

In the data processing stage, ten negative images and ten positive images are removed from the dataset and moved to the **model/data/test/** directory. This data is not seen by the network during the training process, and is used by applications.

To ensure your model gets the same results, please use the same test images. By default HIAS compatible projects will be expecting the same test images.  You can also try with your own image selection, however results may vary and you will need to make additional changes to our HIAS compatible projects.

To specify which test images to use modify the [configuration/config.json](../configuration/config.json) file as shown below:

```
"test_data": [
    "Covid (1128).png",
    "Covid (1183).png",
    "Covid (1239).png",
    "Covid (329).png",
    "Covid (371).png",
    "Covid (55).png",
    "Covid (552).png",
    "Covid (7).png",
    "Covid (842).png",
    "Covid (89).png",
    "Non-Covid (1114).png",
    "Non-Covid (1164).png",
    "Non-Covid (1205).png",
    "Non-Covid (217).png",
    "Non-Covid (457).png",
    "Non-Covid (56).png",
    "Non-Covid (6).png",
    "Non-Covid (715).png",
    "Non-Covid (822).png",
    "Non-Covid (955).png"
],
```

&nbsp;

# Configuration
[configuration/config.json](../configuration/config.json "configuration/config.json")  holds the configuration for our application.

- Change **agent->cores** to the number of cores you have on your training computer.

<details><summary><b>View file contents</b></summary>
<p>

```
{
    "agent": {
        "cores": 8,
        "params": [
            "train",
            "classify",
            "server",
            "classify_http"
        ]
    },
    "data": {
        "dim": 224,
        "file_type": ".png",
        "labels": [0, 1],
        "test": "model/data/test",
        "test_data": [
            "Covid (1128).png",
            "Covid (1183).png",
            "Covid (1239).png",
            "Covid (329).png",
            "Covid (371).png",
            "Covid (55).png",
            "Covid (552).png",
            "Covid (7).png",
            "Covid (842).png",
            "Covid (89).png",
            "Non-Covid (1114).png",
            "Non-Covid (1164).png",
            "Non-Covid (1205).png",
            "Non-Covid (217).png",
            "Non-Covid (457).png",
            "Non-Covid (56).png",
            "Non-Covid (6).png",
            "Non-Covid (715).png",
            "Non-Covid (822).png",
            "Non-Covid (955).png"

        ],
        "train_dir": "model/data/train",
        "valid_types": [
            ".JPG",
            ".JPEG",
            ".PNG",
            ".GIF",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif"
        ]
    },
    "model": {
        "features": "model/features",
        "x_feature": "model/features/data_df_X_train_lite.csv",
        "y_feature": "model/features/data_df_y_train_lite.csv"
    }
}
```

</p>
</details><br />

The configuration object contains 3 Json Objects (agent, data and model). Agent has the information used to set up your server, data has the configuration related to preparing the training and validation data and model holds the model file paths.

&nbsp;

# Training
Now you are ready to train your model.

## Metrics
We can use metrics to measure the effectiveness of our model. In this network you will use the following metrics:

- Accuracy
- Precision
- Recall
- F1

These metrics will be displayed and plotted once our model is trained.

## Start Training
Ensuring you have completed all previous steps, you can start training using the following command.

```
python agent.py train
```

This tells the application to start training the model.

## Training Data
First the training and validation data will be prepared. The program will first move the test data specified in the configuration to the **model/data/test** directory.

```
2021-05-23 03:29:56,150 - Agent - INFO - model/data/train/Covid (1128).png moved to model/data/test/Covid (1128).png
2021-05-23 03:29:56,150 - Agent - INFO - model/data/train/Covid (1183).png moved to model/data/test/Covid (1183).png
2021-05-23 03:29:56,150 - Agent - INFO - model/data/train/Covid (1239).png moved to model/data/test/Covid (1239).png
2021-05-23 03:29:56,150 - Agent - INFO - model/data/train/Covid (329).png moved to model/data/test/Covid (329).png
2021-05-23 03:29:56,150 - Agent - INFO - model/data/train/Covid (371).png moved to model/data/test/Covid (371).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Covid (55).png moved to model/data/test/Covid (55).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Covid (552).png moved to model/data/test/Covid (552).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Covid (7).png moved to model/data/test/Covid (7).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Covid (842).png moved to model/data/test/Covid (842).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Covid (89).png moved to model/data/test/Covid (89).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Non-Covid (1114).png moved to model/data/test/Non-Covid (1114).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Non-Covid (1164).png moved to model/data/test/Non-Covid (1164).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Non-Covid (1205).png moved to model/data/test/Non-Covid (1205).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Non-Covid (217).png moved to model/data/test/Non-Covid (217).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Non-Covid (457).png moved to model/data/test/Non-Covid (457).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Non-Covid (56).png moved to model/data/test/Non-Covid (56).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Non-Covid (6).png moved to model/data/test/Non-Covid (6).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Non-Covid (715).png moved to model/data/test/Non-Covid (715).png
2021-05-23 03:29:56,151 - Agent - INFO - model/data/train/Non-Covid (822).png moved to model/data/test/Non-Covid (822).png
2021-05-23 03:29:56,152 - Agent - INFO - model/data/train/Non-Covid (955).png moved to model/data/test/Non-Covid (955).png
```
Next the training data will be prepared. You will see file name and label for each sample before the

```
2021-05-23 03:40:53,078 - Agent - INFO - Negative data: 1219
2021-05-23 03:40:53,078 - Agent - INFO - Positive data: 1242
2021-05-23 03:40:53,078 - Agent - INFO - Batch: (2461, 4096)
2021-05-23 03:40:53,078 - Agent - INFO - Labels: (2461,)
2021-05-23 03:40:53,087 - Agent - INFO - Training data: (1968, 4096)
2021-05-23 03:40:53,087 - Agent - INFO - Training labels: (1968,)
2021-05-23 03:40:53,087 - Agent - INFO - Validation data: (493, 4096)
2021-05-23 03:40:53,087 - Agent - INFO - Validation labels: (493,)
2021-05-23 03:40:57,618 - Agent - INFO - Data preperation complete.
```

### Model Summary

Before the model begins training, you will be shown the model summary for the [VGG19 Model](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py) that is used for extracting features and data points.

```
2021-05-23 03:41:48,932 - Agent - INFO - X train: (1968, 4096)
2021-05-23 03:41:48,932 - Agent - INFO - Y train: (1968, 2)
2021-05-23 03:42:27,242 - Agent - INFO - Loaded xDNN Model.
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312
=================================================================
Total params: 139,570,240
Trainable params: 139,570,240
Non-trainable params: 0
_________________________________________________________________
2021-05-23 03:42:27,245 - Agent - INFO - VGG19 Model loaded
```

## Training Results
Below are the training results for 150 epochs.

<img src="../model/plots/accuracy_chart.png" alt="Adam Optimizer Results" />

_Fig 1. Accuracy_

<img src="../model/plots/metrics.png" alt="Loss" />

_Fig 2. Precision, Recall and F1_

<img src="../model/plots/confusion_matrix.png" alt="AUC" />

_Fig 3. Confusion Matrix_

```
2021-05-23 03:45:30,042 - Agent - INFO - ###################### Results ####################
2021-05-23 03:45:30,042 - Agent - INFO - Time: 139.48seconds
2021-05-23 03:45:30,043 - Agent - INFO - Accuracy: 0.9675456389452333
2021-05-23 03:45:30,044 - Agent - INFO - Precision: 0.9675456389452333
2021-05-23 03:45:30,044 - Agent - INFO - Recall: 0.9675456389452333
2021-05-23 03:45:30,045 - Agent - INFO - F1 score: 0.9675456389452332
2021-05-23 03:45:30,047 - Agent - INFO - Cohens kappa: 0.9350182884634395

2021-05-23 03:45:30,333 - Agent - INFO - Confusion Matrix: [[230  10] [  6 247]]
2021-05-23 03:45:30,333 - Agent - INFO - Normalized Confusion Matrix: [[0.95833333 0.04166667] [0.02371542 0.97628458]]

2021-05-23 03:45:30,490 - Agent - INFO - True Positives: 247(50.10141987829615%)
2021-05-23 03:45:30,491 - Agent - INFO - False Positives: 10(2.028397565922921%)
2021-05-23 03:45:30,491 - Agent - INFO - True Negatives: 230(46.65314401622718%)
2021-05-23 03:45:30,491 - Agent - INFO - False Negatives: 6(1.2170385395537526%)
2021-05-23 03:45:30,491 - Agent - INFO - Specificity: 0.9583333333333334
2021-05-23 03:45:30,491 - Agent - INFO - Misclassification: 16(3.2454361054766734%)
```

## Metrics Overview
| Accuracy | Recall | Precision | F1 |
| ---------- | ---------- | ---------- | ---------- |
| 0.9675456389452333 | 0.9675456389452333 | 0.9675456389452333 | 0.9675456389452332 |


## Metrics
| Figures of merit     | Amount/Value | Percentage |
| -------------------- | ----- | ---------- |
| True Positives       | 247 | 50.10141987829615% |
| False Positives      | 10 | 2.028397565922921% |
| True Negatives       | 230 | 46.65314401622718% |
| False Negatives      | 6 | 1.2170385395537526% |
| Misclassification    | 16 | 3.2454361054766734% |
| Sensitivity / Recall |  0.9675456389452333  | 97% |
| Specificity          | 0.9583333333333334  | 96% |

&nbsp;

# Testing

Now you will test the classifier on your training machine. You will use the 20 images that were removed from the training data in a previous part of this tutorial.

To run the AI Agent in test mode use the following command:

```
python3 agenty.py classify
```

You should see the following which shows you the VGG19 network architecture:

```
Model: "SarsCov2xDNN"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312
=================================================================
Total params: 139,570,240
Trainable params: 139,570,240
Non-trainable params: 0
_________________________________________________________________
2021-05-23 04:07:51,909 - Agent - INFO - VGG19 Model loaded
```

Finally the application will start processing the test images and the results will be displayed in the console.

<details><summary><b>View output</b></summary>
<p>

    2021-05-23 04:07:52,530 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Positive) in 0.2735898494720459 seconds.
    2021-05-23 04:07:53,012 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Negative) in 0.2899034023284912 seconds.
    2021-05-23 04:07:53,479 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Positive) in 0.27687501907348633 seconds.
    2021-05-23 04:07:53,956 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Positive) in 0.2845020294189453 seconds.
    2021-05-23 04:07:54,426 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Negative) in 0.27788805961608887 seconds.
    2021-05-23 04:07:54,904 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Negative) in 0.28234362602233887 seconds.
    2021-05-23 04:07:55,384 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Negative) in 0.28928375244140625 seconds.
    2021-05-23 04:07:55,859 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Positive) in 0.2810978889465332 seconds.
    2021-05-23 04:07:56,342 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Negative) in 0.29034996032714844 seconds.
    2021-05-23 04:07:56,813 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Positive) in 0.27492523193359375 seconds.
    2021-05-23 04:07:57,303 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Positive) in 0.29724907875061035 seconds.
    2021-05-23 04:07:57,784 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Positive) in 0.28563737869262695 seconds.
    2021-05-23 04:07:58,248 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Positive) in 0.27115321159362793 seconds.
    2021-05-23 04:07:58,735 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Negative) in 0.2935960292816162 seconds.
    2021-05-23 04:07:59,204 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Positive) in 0.2771143913269043 seconds.
    2021-05-23 04:07:59,682 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Positive) in 0.2868936061859131 seconds.
    2021-05-23 04:08:00,163 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Negative) in 0.28818488121032715 seconds.
    2021-05-23 04:08:00,656 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Negative) in 0.2862370014190674 seconds.
    2021-05-23 04:08:01,146 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Negative) in 0.29833269119262695 seconds.
    2021-05-23 04:08:01,616 - Agent - INFO - SARS-CoV-2 xDNN Classifier correctly detected (True Negative) in 0.27336955070495605 seconds.
    2021-05-23 04:08:01,616 - Agent - INFO - Images Classified: 20
    2021-05-23 04:08:01,616 - Agent - INFO - True Positives: 10
    2021-05-23 04:08:01,616 - Agent - INFO - False Positives: 0
    2021-05-23 04:08:01,616 - Agent - INFO - True Negatives: 10
    2021-05-23 04:08:01,617 - Agent - INFO - False Negatives: 0
    2021-05-23 04:08:01,617 - Agent - INFO - Total Time Taken: 5.6785266399383545
</p>
</details><br />

In the current terminal, now use the following command:

```
python3 agenty.py server
```

This will start the server on your training machine that exposes the model via a REST API. Now open a new terminal, navigate to the project root and use the following command:

```
python3 agenty.py classify_http
```

This will start agent in HTTP Inference mode. The agent will loop through the testing data and send each image to the server for classification, the results are then displayed in the console.

<details><summary><b>View output</b></summary>
<p>

    2021-05-23 04:13:24,168 - Agent - INFO - Helpers class initialization complete.
    2021-05-23 04:13:24,168 - Agent - INFO - Agent initialization complete.
    2021-05-23 04:13:24,168 - Agent - INFO - Data class initialization complete.
    2021-05-23 04:13:24,168 - Agent - INFO - Model class initialization complete.

    2021-05-23 04:13:24,168 - Agent - INFO - Sending request for: model/data/test/Covid (1128).png

    2021-05-23 04:13:24,818 - Agent - INFO - SARS-CoV-2 correctly detected (True Positive) in 0.6492893695831299 seconds.

    2021-05-23 04:13:24,818 - Agent - INFO - Sending request for: model/data/test/Non-Covid (1114).png

    2021-05-23 04:13:25,307 - Agent - INFO - SARS-CoV-2 correctly detected (True Negative) in 0.4895467758178711 seconds.

    2021-05-23 04:13:25,308 - Agent - INFO - Sending request for: model/data/test/Covid (371).png

    2021-05-23 04:13:25,788 - Agent - INFO - SARS-CoV-2 correctly detected (True Positive) in 0.4803190231323242 seconds.

    2021-05-23 04:13:25,788 - Agent - INFO - Sending request for: model/data/test/Covid (1183).png

    2021-05-23 04:13:26,381 - Agent - INFO - SARS-CoV-2 correctly detected (True Positive) in 0.5924265384674072 seconds.

    2021-05-23 04:13:26,381 - Agent - INFO - Sending request for: model/data/test/Non-Covid (56).png

    2021-05-23 04:13:26,862 - Agent - INFO - SARS-CoV-2 correctly detected (True Negative) in 0.48140954971313477 seconds.

    2021-05-23 04:13:26,862 - Agent - INFO - Sending request for: model/data/test/Non-Covid (6).png

    2021-05-23 04:13:27,354 - Agent - INFO - SARS-CoV-2 correctly detected (True Negative) in 0.49121570587158203 seconds.

    2021-05-23 04:13:27,354 - Agent - INFO - Sending request for: model/data/test/Non-Covid (1205).png

    2021-05-23 04:13:27,848 - Agent - INFO - SARS-CoV-2 correctly detected (True Negative) in 0.4936649799346924 seconds.

    2021-05-23 04:13:27,848 - Agent - INFO - Sending request for: model/data/test/Covid (329).png

    2021-05-23 04:13:28,339 - Agent - INFO - SARS-CoV-2 correctly detected (True Positive) in 0.49178147315979004 seconds.

    2021-05-23 04:13:28,340 - Agent - INFO - Sending request for: model/data/test/Non-Covid (822).png

    2021-05-23 04:13:28,835 - Agent - INFO - SARS-CoV-2 correctly detected (True Negative) in 0.4957602024078369 seconds.

    2021-05-23 04:13:28,836 - Agent - INFO - Sending request for: model/data/test/Covid (55).png

    2021-05-23 04:13:29,324 - Agent - INFO - SARS-CoV-2 correctly detected (True Positive) in 0.48885297775268555 seconds.

    2021-05-23 04:13:29,325 - Agent - INFO - Sending request for: model/data/test/Covid (1239).png

    2021-05-23 04:13:29,821 - Agent - INFO - SARS-CoV-2 correctly detected (True Positive) in 0.4962484836578369 seconds.

    2021-05-23 04:13:29,821 - Agent - INFO - Sending request for: model/data/test/Covid (552).png

    2021-05-23 04:13:30,306 - Agent - INFO - SARS-CoV-2 correctly detected (True Positive) in 0.48441243171691895 seconds.

    2021-05-23 04:13:30,306 - Agent - INFO - Sending request for: model/data/test/Covid (842).png

    2021-05-23 04:13:30,783 - Agent - INFO - SARS-CoV-2 correctly detected (True Positive) in 0.4772377014160156 seconds.

    2021-05-23 04:13:30,783 - Agent - INFO - Sending request for: model/data/test/Non-Covid (715).png

    2021-05-23 04:13:31,290 - Agent - INFO - SARS-CoV-2 correctly detected (True Negative) in 0.5065999031066895 seconds.

    2021-05-23 04:13:31,290 - Agent - INFO - Sending request for: model/data/test/Covid (7).png

    2021-05-23 04:13:31,767 - Agent - INFO - SARS-CoV-2 correctly detected (True Positive) in 0.47717905044555664 seconds.

    2021-05-23 04:13:31,767 - Agent - INFO - Sending request for: model/data/test/Covid (89).png

    2021-05-23 04:13:32,265 - Agent - INFO - SARS-CoV-2 correctly detected (True Positive) in 0.4975433349609375 seconds.

    2021-05-23 04:13:32,265 - Agent - INFO - Sending request for: model/data/test/Non-Covid (217).png

    2021-05-23 04:13:32,753 - Agent - INFO - SARS-CoV-2 correctly detected (True Negative) in 0.48796725273132324 seconds.

    2021-05-23 04:13:32,753 - Agent - INFO - Sending request for: model/data/test/Non-Covid (457).png

    2021-05-23 04:13:33,244 - Agent - INFO - SARS-CoV-2 correctly detected (True Negative) in 0.49101877212524414 seconds.

    2021-05-23 04:13:33,249 - Agent - INFO - Sending request for: model/data/test/Non-Covid (955).png

    2021-05-23 04:13:33,743 - Agent - INFO - SARS-CoV-2 correctly detected (True Negative) in 0.4938371181488037 seconds.

    2021-05-23 04:13:33,743 - Agent - INFO - Sending request for: model/data/test/Non-Covid (1164).png

    2021-05-23 04:13:34,231 - Agent - INFO - SARS-CoV-2 correctly detected (True Negative) in 0.488292932510376 seconds.

    2021-05-23 04:13:34,231 - Agent - INFO - Images Classified: 20
    2021-05-23 04:13:34,231 - Agent - INFO - True Positives: 10
    2021-05-23 04:13:34,231 - Agent - INFO - False Positives: 0
    2021-05-23 04:13:34,232 - Agent - INFO - True Negatives: 10
    2021-05-23 04:13:34,232 - Agent - INFO - False Negatives: 0
    2021-05-23 04:13:34,232 - Agent - INFO - Total Time Taken: 10.054603576660156

</p>
</details><br />

&nbsp;

# HIAS UI

Now that your classifier is setup and running, you can interact with it via the HIAS UI, and from other HIAS integrated applications. Before you can do so there is a final step to take on your server.

![HIAS AI Inference](../assets/images/hias-ai-inference-endpoint.jpg)

Head to the AI Agent page for your classifier on HIAS **(AI->Agents->List->Your Agent)**. On the edit page you will see the **Inference Endpoint**, you need to copy that value.

## NGINX

Now in console open the NGINX config file:

```
sudo nano /etc/nginx/sites-available/default
```
Find **ADD NEW ENDPOINTS AFTER THIS NOTICE**, and add the following, replacing **YourEndpoint** with your inference endpoint value, and  **YourIp/YourPort** with the IP/port of your device.
```
location ~ ^/AI/YourEndpoint/(.*)$ {
                auth_basic "Restricted";
                auth_basic_user_file /etc/nginx/security/htpasswd;
                proxy_pass http://YourIp:YourPort/$1;
            }
```
Save the file and exit, then run the following command:

```
sudo systemctl reload nginx
```

## Inference

Now you are set up to communicate with the SARS-CoV-2 xDNN Classifier from HIAS. Head to **(AI->Agents->List)** and then click on the **Inference** link.

![HIAS AI Inference](../assets/images/hias-ai-inference.jpg)

Once on the Inference page upload the twenty test images. Now make sure the server is running on the RPI and click the data to send it to the SARS-CoV-2 xDNN Classifier for classification.

### Verification

As we know from the filenames in advance whether an image is negative or positive, we can compare the classification with the file name to check if a classification is a true/false positive, or a true/false negative. In the Diagnosis Results area Diagnosis represents the classification provided by the SARS-CoV-2 xDNN Classifier, and Result provides the verification result. You should get the same results as when testing earlier back in the tutorial. The UI should 1 false negative and one false positive.

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and youlcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Nitin Mane](https://www.leukemiaairesearch.com/association/volunteers/nitin-mane "Nitin Mane") - [Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss") Deep Learning, Aurangabad, India

&nbsp;

# Versioning

You use SemVer for versioning. For the versions available, see [Releases](../releases "Releases").

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues

You use the [repo issues](../issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.
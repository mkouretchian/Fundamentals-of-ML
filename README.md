# Fundamentals-of-ML
This repo is for projects our team have done in Fundamentals of ML Fall 2018 at UF. 


# Handwritten Character Recognition:
In this project, we wrote an script to transform images into binary format. Then diagonalized version of KNN developed by Kumar et al[1].

Dependencies Install Opencv software in the environment. This is necessary to convert the binary images into contour images. The current implementation is tested under Opencv 3.4.2 which could be downloaded through the following link: https://opencv.org/releases.html

There are six main files associated with this project: classification.py, feature_extraction.py, findContours.py, normalization.py, train.py, and test.py. The only file that requires running is test.py, all other desired functionality from the previous files mentioned will be imported and used appropriately. While running this code it is essential to include a folder named "Traning_Images" without the quotation marks to store all of the images for testing after normalization. This folder is used to prevent clutter within the directory, but this can also be changed within the normalization.py file. However, if the directory is changed it will also need to be changed in normalization.py.


# KNN and Probabilistic Generative Model:
Comparison of Supervised methods on data with different dimensions:
Datasets
Each category of dataset includes separate training and hold out test data.

2 Dimension data
7 Dimension data
HyperSpectral data
hw02.py includes implementation of KNN and probabilistic generative model(regular and diagonal versions) from scratch. After defining the models, K-Fold cross validation has been used for parameter tunning using the validation data. Finally the model with best performance for each category has been used to predict the test labels properly.

# Object Detection in Imagery Data:

These are the libraries I am using: import matplotlib.pyplot as plt import numpy as np from sklearn.neighbors import KNeighborsClassifier from scipy import stats from random import randint from sklearn.metrics import accuracy_score, confusion_matrix, classification_report import itertools

The project 00 includes three separate train, test and run.py In the train.py I will use the cross validation along with the training function to tune the KNN parameters so I could use them later in run.py. the range of K-Values could be changes easily.

Run.py: At the beginning I will read the bounding box from the more_red_cars.npy to come up with the all centers of the cars. I will use this as my ground truth which later expanded in the run.py to the 5x5 windows around it.

After marking the intances of ground truth, the random points from Training data and ground truth will be generated to form training and validation dataset as we could not use random functions such as train-test-split. Test data will be formed from random points in the image data.

After generating the data, the prediction function (KNeighbor_Train) would be called to make the predictions on validation dataset. Confusion matrix has been plotted using the instructions on [2]. The plot to show the accuracy over the range of different K has been plotted too. I have commented the code to make image from data points of the cars.

The cross validation function has been defined in Train.py to get the best parameter for K.

Running the scripts: I have tested in both terminal and spyder IDE undr the Anaconda.

Terminal: python run.py would do everything for you.

P.S. the .npy data files are the empty ones intially was posted for the project. Due to file size limitation for pushing to the repo, I did not push them. Therefore, my train.py , test.py and run.py and more_red_cars.npy should be pulled to the repo with the original data. Otherwise, it will give the index error for reading the training data.



1.Kumar, M., Jindal, M.K. and Sharma, R.K., 2011, November. k-nearest neighbor based offline handwritten Gurmukhi character recognition. In 2011 International Conference on Image Information Processing (pp. 1-4). IEEE.
2.http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

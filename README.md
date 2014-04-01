NeuralNetworks
==============
This project implements different classiffcation methods and compare its performance
in classifying handwritten digits. The following objectives are met:
 How Neural Network works and use Feed Forward, Back Propagation to implement Neural Network
 How K-Nearest Neighbors can be used for classication task

The dataset for handwritten digits is taken from http://www.cs.nyu.edu/~roweis/data.html
The MNIST dataset consists of a training set of 60000 examples and test set of 10000 examples. All
digits have been size-normalized and centered in a xed image of 2828 size. In original dataset, each pixel
in the image is represented by an integer between 0 and 255, where 0 is black, 255 is white and anything
between represents dierent shade of gray.

 preprocess.m: performs some preprocess tasks, and outputs the preprocessed train, validation and test
                data with their corresponding labels (1 of k encoding scheme for error calculation)
 script.m: Main Matlab script that calls other functions (Neural network implementation and KNN)
 sigmoid.m: Calculates sigmoid 
 nnObjFucntion.m: computes the error function of Neural Network.
 nnPredict.m: predicts the label of data given the parameters of Neural Network.
 initializeWeights.m: return the random weights for Neural Network given the number of node in the
                input layer and output layer.
 fmincg.m: perform optimization task by using conjugate gradient descent (Used as is)
 knnPredict.m: knnPredict predicts the label of given data by using k-nearest neighbor classication
                algorithm.

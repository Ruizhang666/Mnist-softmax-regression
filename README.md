# KMNIST Dataset Classification

## Introduction

In this project, we propose utilizing a one-layer neural network with logistic regression as the activation function to classify two-class KMNIST datasets and SoftMax Regression to classify multi-class KMNIST datasets. We will experiment with discerning between two classes of data (class 2 vs class 6, class 0 vs class 6), and with multi-class classification. We first implement logistic regression for the two-class classification. Softmax regression is the generalization of logistic regression for multiple class classification, so we also use SoftMax to classify class 0-9. We apply K-fold cross validation and early stopping technique to prevent overfitting. We also utilize mini batches for stochastic gradient descent in our approach, which makes the algorithm far more efficient than the one that is implemented with original gradient descent. 

We experiment with different combinations of hyperparameters, specifically we pay attention to the learning rate, normalization method and early stopping method. We find that with appropriate learning rate, we could boost the class 0 vs class 2 classification test accuracy from % to %. As a result, for class 0 vs class 6 classification, we are able to achieve a test accuracy of 98.34%. We obtain a 87.95% test accuracy for class 2 vs class 6. We also achieve a 70.34% accuracy on class 0-9 classification with early stopping and appropriate hyper parameters.

## Instructions on How to Run the Code
- network.py<br />
We include the code for buildling the network model in the network.py
- data.py<br />
We include the code for transforming data in the data.py
- final version.ipynb<br />
We mainly use this notebook for our modelling, visualizations and hyperparameter tunings.

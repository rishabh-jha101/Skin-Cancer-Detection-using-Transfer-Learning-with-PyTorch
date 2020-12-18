# Skin-Cancer-Detection-using-Transfer-Learning-with-PyTorch
Benign Vs Malignant Skin Cancer Detection using Transfer Learning with VGG16 architecture.

## Table of Contents
* [General Information](#general-info)
* [Architecture](#architecture)
* [Technologies](#technologies)
* [Setup](#setup)
* [Dataset](#dataset)
* [Result](#result)

## General Information
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes.
### Transfer Learning
![picture alt](https://miro.medium.com/max/798/1*ZkPBqU8vx2vAgcLpz9pi5g.jpeg)
1. Following is the general outline for transfer learning for object recognition:
2. Load in a pre-trained CNN model trained on a large dataset
3. Freeze parameters (weights) in model’s lower convolutional layers
4. Add custom classifier with several layers of trainable parameters to model
5. Train classifier layers on training data available for task
6. Fine-tune hyperparameters and unfreeze more layers as needed

## Architecture
![picture alt](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

In this classification problem the, linear fully connected layers of VGG16 were changed. Number of hidden nodes were changed to suit the limited data at hand. The last layer with 1000 nodes was replaced with 2 nodes. The weights and stride of kernels are kept same/frozen and only parameters in the linear layers were allowed to be changed.
In this way high generalization was achieved while avoiding overfitting with comparatively less data.

## Technologies
Project is created with:
* Python: 3.7.4
* PyTorch: 1.7.1
* torchvision: 0.8.2

## Setup
### Hyperparameters
learning rate = 0.001
batch_size = 20
epochs = 10

To train the model, run this code in the directory
`python train.py`
& to test
`python test.py`

## Dataset
### kaggle
https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign

## Result
 Minimum validation error was achieved after a few epochs.
 
 Model trained with Overall 93% accuracy with 
 Test Loss: 0.017195
 
 ### Test Accuracy for each class
 Test Accuracy of Benign: 92% (175/189)
 Test Accuracy of Malignant: 94% (132/140)

 Test Accuracy (Overall): 93% (307/329)

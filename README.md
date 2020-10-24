# Building an AI COVID-19 Product from Scratch with Pytorch (Implementation Time: Under 2 hours)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1H8SbSmrI3jjGL_8lWy1fPSeHsuL_Lp60)

## Background:

In screening for Covid-19, patients can first be screened for flu-like symptoms using nasal swap to confirm their COVID-19 status. After 14 days of quarantine for confirmed cases, the hospital draws the patient’s blood and takes the patient’s chest X-ray. Chest X-ray is a golden standard for physicians and radiologists to check for the infection caused by the virus. An x-ray imaging will allow your doctor to see your lungs, heart and blood vessels to help determine if you have pneumonia. When interpreting the x-ray, the radiologist will look for white spots in the lungs (called infiltrates) that identify an infection. This exam, together with other vital signs such as temperature, or flu-like symptoms, will also help doctors determine whether a patient is infected with COVID-19 or other pneumonia-related diseases. The standard procedure of pneumonia diagnosis involves a radiologist reviewing chest x-ray images and send the result report to a patient’s primary care physician (PCP), who then will discuss the results with the patient.

 ![Alt text](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection/blob/main/Images/Fig1-Concept-idea.jpg)
 
 _Fig 1: Current chest X-ray diagnosis vs. novel process with PneumoScan.ai_


A survey by the University of Michigan shows that patients usually expect the result came back after 2-3 days a chest X-ray test for pneumonia. (Crist, 2017) However, the average wait time for the patients is 11 days (2 weeks). This long delay happens because radiologists usually need at least 20 minutes to review the X-ray while the number of images keeps stacking up after each operation day of the clinic. New research has found that an artificial intelligence (AI) radiology platform such as our CovidScan.ai can dramatically reduce the patient’s wait time significantly, cutting the average delay from 11 days to less than 3 days for abnormal radiographs with critical findings. (Mauro et al., 2019) With this wait-tine reduction, patients I critical cases will receive their results faster, and receive appropriate care sooner. 

 ![Alt text](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection/blob/main/Images/Fig2-AI-vs-Manual.png)
 
_Fig 2: Chart of wait-time reduction of AI radiology tool (data from a simulation stud reported in Mauro et al., 2019)._

In this tutorial, we’ll show you how to use Pytorch to build a machine learning web application to classify whether a patient has COVID-19, Pneumonia or no sign of any infection (normal) from chest x-ray images. We will focus on the Pytorch component of the AI application. 

Below are the 4 main step we’ll go over in the tutorial:

1.	Collecting Dataset

2. Preprocessing the Data

3.	Building and Training the Deep Learning Model

4.	Evaluating the Model

5.	Deploying the Model a Web-app 



## 1.	Collecting the Data:
To build the chest X-ray detection models, we used combined 2 sources of dataset:
1.	The first source is the RSNA Pneumonia Detection Challenge dataset available on Kaggle contains several deidentified CXRs with 2 class labels of pneumonia and normal.
2.	The COVID-19 image data collection repository on GitHub is a growing collection of deidentified CXRs from COVID-19 cases internationally. The data is collected by Joseph Paul Cohen and his fellow collaborators at the University of Montreal
Eventually, our dataset consists of 5433 training data points, 624 validation data points and 16 test data points.

## 2. Preprocessing the data:
First, we import all the require package:
```
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import PIL
import scipy.ndimage as nd
```

The more data, the better the model will learn. Hence, apply some data augmentation to generate different variations of the original data to increase the sample size for training, validation and testing process. This augmentation can be performed by defining a set of transforming functions in the torchvision module. The detailed codes are as following:

```
#Using the transforms module in the torchvision module, we define a set of functions that perform data augmentation on our dataset to obtain more data.#
transformers = {'train_transforms' : transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),
'test_transforms' : transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),
'valid_transforms' : transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])}
trans = ['train_transforms','valid_transforms','test_transforms']
path = "/content/gdrive/My Drive/chest_xray/"
categories = ['train','val','test']
```
After defining the transformers, now we can use torchvision.datasets.ImageFolder module we load images from our dataset directory and apply the predefined transformers on them as following:
```
dset = {x : torchvision.datasets.ImageFolder(path+x, transform=transformers[y]) for x,y in zip(categories, trans)}
dataset_sizes = {x : len(dset[x]) for x in ["train","test"]}
num_threads = 4

#By passing a dataset instance into a DataLoader module, we create dataloader which generates images in batches.
dataloaders =  {x : torch.utils.data.DataLoader(dset[x], batch_size=256, shuffle=True, num_workers=num_threads)
               for x in categories}
```
To make sure we are loading the data correctly and the augmentation is performed, let’s check some generated data using matplotlib and numpy using this block of code:
```
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std*inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
inputs,classes = next(iter(dataloaders["train"]))
out = torchvision.utils.make_grid(inputs)
class_names = dataset["train"].classes
imshow(out, title = [class_names[x] for x in classes])
```
From the plot of a batch of sample images, we can see the data is loaded properly and augmented in different variations. Then, we can start our model building process.

## 2. Building the Model:

For the project, we use the pretrained ResNet 152 provided in Pytorch libary. ResNet models is arranged in a series of convolutional layers in very deep network architecture. The layers are in form of residual blocks, which allow gradient flow in very deep networks using skip connections as shown in fig. These connections help preventing the problem of vanishing gradients which are very pervasive in very deep convolutional networks. In the last layer of the Resnet, we use the Global Average Pooling layer instead of fully connected layers to reduce the number of parameters created by fully-connected layers to zero. Hence, we can avoid over-fitting (which is a common problem of deep network architecture as Resnet).  

 ![Alt text](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection/blob/main/Images/deep%20network.png)

 
Before we get into the actually model building process, you can refresh you memory on the basic of deep learning using these two recommended tutorial from Pytorch.
Basic Steps in Model building: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
Transfer Learning in Imaging: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
After refreshing your memory on the basic, we can start with this project using the COVID chest X-ray data. First, we need to initalize our model class by calling the nn.Module, which create a graph-like structure of our network. In order to predict well the classes of an image, the neural network needs to be super efficient in extracting the features from the input images. Hence, the model first needs to be trained on a huge dataset to get really good at feature-extraction. However, not everyone, especially beginners in ML, access to powerful GPU or the indept knowledge to train on such big data. That is why we leverage transfer learning in our model building process, which save us  a lot of time and trouble in building an state-of-art model from scratch. Luckily for us, the torchvision module already includes several state of the art models trained on the huge dataset of Imagenet (more than 14 millions of 20,000 categories). Hence, these pretrained model is crazily good at feature extraction of thousand type of objects. In particularly, as we mentioned earlier, the pretrained model of Resnet152 was used in our training process. This transfer learning give us a big advantage in retraining on Hence, we need to define our ResNet-152 in the init of nn.Module for transfer learning.  Pytorch provides us with the ability to take and freeze these powerful feature extractors, attach our own classifiers depending on our problem domain and train the resulting model to suit our problem.

```
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #obtain the ResNet model from torchvision.model library
        self.model = torchvision.models.resnet152(pretrained=True)
        #build our classifier and since we are classifying the images into NORMAL and PNEMONIA, we output a two-dimensional tensor.
        self.classifier = nn.Sequential(
        nn.Linear(self.model.fc.in_features,2),
        nn.LogSoftmax(dim=1))
        #Requires_grad = False denies the ResNet model the ability to update its parameters hence make it unable to train.
        for params in self.model.parameters():
            params.requires_grad = False
            #We replace the fully connected layers of the base model(ResNet model) which served as the classifier with our custom trainable classifier.
        self.model.fc = self.classifier
```


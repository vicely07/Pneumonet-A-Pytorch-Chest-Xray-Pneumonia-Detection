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

**Below are the 4 main step we’ll go over in the tutorial:**

[1.	Collecting Dataset](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#1collecting-the-data)

[2. Preprocessing the Data](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#2-preprocessing-the-data)

[3.	Building the Model](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#3-building-the-model)

[a) Basics of Transfer Learning](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#a-basics-of-transfer-learning)

[b) Architecture of Resnet 152 with Global Average Pooling layer](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#b-architecture-of-resnet-152-with-global-average-pooling-layer)

[c) Retraining Resnet 152 Model in Pytorch](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#c-retraining-resnet-152-model-in-pytorch)

[d) Building the Activation Map For Visualization](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#d-building-the-activation-map-for-visualization)

[4. Developing the Web-app](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#4-developing-the-web-app)

[5. Summary](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#5-summary)



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

## 3. Building the Model:

## a) Basics of Transfer Learning:

In order to predict well the classes of an image, the neural network needs to be super efficient in extracting the features from the input images. Hence, the model first needs to be trained on a huge dataset to get really good at feature-extraction. However, not everyone, especially beginners in ML, access to powerful GPU or the indept knowledge to train on such big data. That is why we leverage transfer learning in our model building process, which save us  a lot of time and trouble in building an state-of-art model from scratch. Luckily for us, the torchvision module already includes several state of the art models trained on the huge dataset of Imagenet (more than 14 millions of 20,000 categories). Hence, these pretrained model is crazily good at feature extraction of thousand type of objects. 

## b) Architecture of Resnet 152 with Global Average Pooling layer:

For the project, we use the pretrained ResNet 152 provided in Pytorch libary. ResNet models is arranged in a series of convolutional layers in very deep network architecture. The layers are in form of residual blocks, which allow gradient flow in very deep networks using skip connections as shown in fig. These connections help preventing the problem of vanishing gradients which are very pervasive in very deep convolutional networks. In the last layer of the Resnet, we use the Global Average Pooling layer instead of fully connected layers to reduce the number of parameters created by fully-connected layers to zero. Hence, we can avoid over-fitting (which is a common problem of deep network architecture as Resnet). 

 ![Alt text](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection/blob/main/Images/deep%20network.png)

 ## c) Retraining Resnet 152 Model in Pytorch:
 
Before we get into the actually model building process, you can refresh you memory on the basic of deep learning using these two recommended tutorial from Pytorch.

Basic Steps in Model building: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

Transfer Learning in Imaging: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

After refreshing your memory on the basic, we can start with this project using the COVID chest X-ray data. First, we need to initalize our model class by calling the nn.Module, which create a graph-like structure of our network. In particularly, as we mentioned earlier, the pretrained model of Resnet152 was used in our training process. This transfer learning give us a big advantage in retraining on Hence, we need to define our ResNet-152 in the init of nn.Module for transfer learning. Then after define the init function, we need to create forward function as part of requirement for Pytorch. 

```
##Build model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torchvision.models.resnet152(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.fc.in_features,2),
            nn.LogSoftmax(dim=1)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.fc = self.classifier
        
    def forward(self, x):
        return self.model(x)
 ```
 Then, we define the fit function within this class. We will actually use the fit function to retrain the Resnet 152 on our data:
 ```
    def fit(self, dataloaders, num_epochs):
        train_on_gpu = torch.cuda.is_available()
        optimizer = optim.Adam(self.model.fc.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, 4)
        criterion = nn.NLLLoss()
        since = time.time()
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc =0.0
        if train_on_gpu:
            self.model = self.model.cuda()
        for epoch in range(1, num_epochs+1):
            print("epoch {}/{}".format(epoch, num_epochs))
            print("-" * 10)
            
            for phase in ['train','test']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()
                
                running_loss = 0.0
                running_corrects = 0.0
                
                for inputs, labels in dataloaders[phase]:
                    if train_on_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print("{} loss:  {:.4f}  acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
                
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
        
        time_elapsed = time.time() - since
        print('time completed: {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 600))
        print("best val acc: {:.4f}".format(best_acc))
        
        self.model.load_state_dict(best_model_wts)
        return self.model
```

## d) Building the Activation Map For Visualization:

We learnt earlier that the last layer of our network is Global Average Pooling layer. This last layer is useful for reducing the a tensor of trained weights from h x w x d to 1 x 1 x d. Then, we calculated the weighted sum from this 1 x 1 x d dimensional tensor and then fed into a softmax function to find the probabilities of the predicted class (COVID-19 or Normal). After getting the confirmed class from the model, we can map back this class to the weighted sum tensor to plot the the class activation map for visualization.

In PyTorch, we can use the register_forward_hook module to obtain activation of the last convolutional layer as described above, we use the  register_forward_hook module. The code is as following:
 ```
class LayerActivations():
    features=[]
    def __init__(self,model):
        self.hooks = []
        #model.layer4 is the last layer of our network before the Global Average Pooling layer(last convolutional layer).
        self.hooks.append(model.layer4.register_forward_hook(self.hook_fn))
        
    def hook_fn(self,module,input,output):
        self.features.append(output)
        
    def remove(self):
        for hook in self.hooks:
            hook.remove()
```
After defining the LayerActivation module, we can use this module to visualize the predicted output on testing set. Hence, for convinience, we define a function called predict_img so we can use this predict and automatically visualize the Activation Map on each images later. The function is defined as following:
```
def predict_img(path, model_ft):
  image_path = path
  img = image_loader(image_path)
  acts = LayerActivations(model_ft)
  img = img.cpu()
  logps = model_ft(img)
  ps = torch.exp(logps) 
  out_features = acts.features[0]
  out_features = torch.squeeze(out_features, dim=0)
  out_features = np.transpose(out_features.cpu(),axes=(1,2,0))
  W = model_ft.fc[0].weight
  top_probs, top_classes = torch.topk(ps, k=2)
  pred = np.argmax(ps.detach().cpu())
  w = W[pred,:]
  cam = np.dot(out_features.cpu(), w.detach().cpu())
  class_activation = nd.zoom(cam, zoom=(32,32),order=1)
  img = img.cpu()
  img = torch.squeeze(img,0)
  img = np.transpose(img,(1,2,0))
  mean = np.array([0.5,0.5,0.5])
  std =  np.array([0.5,0.5,0.5])
  img = img.numpy()
  img = (img + mean) * std
  img = np.clip(img, a_max=1, a_min=0)
  return img, class_activation, pred
```
Now, let's load the testing data folder and use this newly defined predict_img function to visualize all the testing images (with both prediction class and Activation map). The snipet of that code is as following:
```
test_dir='/Test_Set/'
from skimage.io import imread
from PIL import Image
import glob
image_list = []
for filename in glob.glob(test_dir+'/*.jpeg'): 
    #im=Image.open(filename)
    image_list.append(filename)

f, ax = plt.subplots(4,4, figsize=(30,10))

def plot_input(image_list):
  for i in range(len(image_list)):
      img = imread(image_list[i])
      name = image_list[i].split("/")
      name = name[len(name)-1].split(".")[0]
      ax[i//4, i%4].imshow(img, cmap='gray')
      ax[i//4, i%4].set_title(name)
      plt.tight_layout()
  plt.show()
plot_input(image_list)
```

## 4. Developing the Web-app:

As you can see, the final results look really nice. This activation map is super informative for the radiologists to quickly pinpoint the area of infection on chest X-ray. To make our project become more user-frinedly. the final step is web-app development with an interactive UI. The trained weights of the deep learning models are deployed in a form of Django backend web app CovidScan.ai. While the minimal front-end of this web app is done using HTML, CSS, Jquery, Bootstrap. In our latter stage, the web-app will then be deployed and hosted on Debian server. 

## 5. Summary:
In our tutorial, we explore a 5-step deep learning model building process using Pytorch. We also go over the concept of transfer learning and the architecture of our Resnet 152 model. We also learn to visualize the Activation Map using the last layer of our trained network. Eventually, we learnt how this deep neural network is deployed to a web-app.  The detailed web-developmeent process is not in the scope of this tutorial since we focus more on the Pytorch model to make the beginner user understand how we get to the finl visualization output from raw chest X-ray data. If you want to read more on how to implement the web-app, we can read the step-by-step instruction on this github tutorial.

For this project, we only implement a binary classification of 2 classes (COVID-19 and Normal). If you want to get more inspiration on building a AI-based product from scratch with multi-class data using Pytorch and FastAi, you can check out this other project created by our team called HemoCount: https://devpost.com/software/hemonet-an-ai-based-white-blood-cell-count-platform?ref_content=user-portfolio&ref_feature=in_progress

We hope you will have a good start by implementing this award-winning project, and be inspired to join other hackathons or datathon competition to build many other awesome AI products from scratch!




 


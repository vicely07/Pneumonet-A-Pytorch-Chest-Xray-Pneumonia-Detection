# Building an AI COVID-19 Product from Scratch with Pytorch (Implementation Time: Under 2 hours)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1H8SbSmrI3jjGL_8lWy1fPSeHsuL_Lp60)

## Background:
*[![This project won 2nd place at FB AI Hackathon*](data:image/png;base64,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)]

In screening for Covid-19, patients can first be screened for flu-like symptoms using nasal swap to confirm their COVID-19 status. After 14 days of quarantine for confirmed cases, the hospital draws the patient’s blood and takes the patient’s chest X-ray. Chest X-ray is a golden standard for physicians and radiologists to check for the infection caused by the virus. An x-ray imaging will allow your doctor to see your lungs, heart and blood vessels to help determine if you have pneumonia. When interpreting the x-ray, the radiologist will look for white spots in the lungs (called infiltrates) that identify an infection. This exam, together with other vital signs such as temperature, or flu-like symptoms, will also help doctors determine whether a patient is infected with COVID-19 or other pneumonia-related diseases. The standard procedure of pneumonia diagnosis involves a radiologist reviewing chest x-ray images and send the result report to a patient’s primary care physician (PCP), who then will discuss the results with the patient.

 ![Alt text](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection/blob/main/Images/Fig1-Concept-idea.jpg)
 
 _Fig 1: Current chest X-ray diagnosis vs. novel process with PneumoScan.ai_


A survey by the University of Michigan shows that patients usually expect the result came back after 2-3 days a chest X-ray test for pneumonia. (Crist, 2017) However, the average wait time for the patients is 11 days (2 weeks). This long delay happens because radiologists usually need at least 20 minutes to review the X-ray while the number of images keeps stacking up after each operation day of the clinic. New research has found that an artificial intelligence (AI) radiology platform such as our CovidScan.ai can dramatically reduce the patient’s wait time significantly, cutting the average delay from 11 days to less than 3 days for abnormal radiographs with critical findings. (Mauro et al., 2019) With this wait-tine reduction, patients I critical cases will receive their results faster, and receive appropriate care sooner. 

 ![Alt text](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection/blob/main/Images/Fig2-AI-vs-Manual.png)
 
_Fig 2: Chart of wait-time reduction of AI radiology tool (data from a simulation stud reported in Mauro et al., 2019)._

In this tutorial, we’ll show you how to use Pytorch to build a machine learning web application to classify whether a patient has COVID-19, Pneumonia or no sign of any infection (normal) from chest x-ray images. We will focus on the Pytorch component of the AI application. 

**Below are the 4 main step we’ll go over in the tutorial (We also attch the appoximate time that you should spend on reading and implementing the code of each section to understand it throughly):**

[1.	Collecting Dataset (2 minutes)](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#1collecting-the-data)

[2. Preprocessing the Data (10 minutes)](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#2-preprocessing-the-data)

[3.	Building the Model (45 minutes)](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#3-building-the-model)

[a) Basics of Transfer Learning](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#a-basics-of-transfer-learning)

[b) Architecture of Resnet 152 with Global Average Pooling layer](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#b-architecture-of-resnet-152-with-global-average-pooling-layer)

[c) Retraining Resnet 152 Model in Pytorch](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#c-retraining-resnet-152-model-in-pytorch)

[d) Building the Activation Map For Visualization](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#d-building-the-activation-map-for-visualization)

[4. Developing the Web-app (30 minutes)](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#4-developing-the-web-app)

[5. Summary & Additional Resorces (5 minutes)](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection#5-summary)

[References](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection/blob/main/README.md#references)


## 1.	Collecting the Data (2 minutes):
To build the chest X-ray detection models, we used combined 2 sources of dataset:
1.	The first source is the RSNA Pneumonia Detection Challenge dataset available on Kaggle contains several deidentified CXRs with 2 class labels of pneumonia and normal.
2.	The COVID-19 image data collection repository on GitHub is a growing collection of deidentified CXRs from COVID-19 cases internationally. The data is collected by Joseph Paul Cohen and his fellow collaborators at the University of Montreal
Eventually, our dataset consists of 5433 training data points, 624 validation data points and 16 test data points.

## 2. Preprocessing the data (10 minutes):
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

## 3. Building the Model (45 minutes):

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

def predict_image(image, model_ft):
  img, class_activation, pred = predict_img(image, model_ft)
  print(pred.item())
  name = image.split("/")
  name = name[len(name)-1].split(".")[0]
  img = Image.fromarray((img * 255).astype(np.uint8))
  plt.ioff()
  plt.imshow(class_activation, cmap='jet',alpha=1)
  plt.imshow(img, alpha=0.55)
  plt.title(dset['test'].classes[pred])
  plt.tight_layout()

  # plt.show()

predict_image(image_list[12], model_ft)
```

## 4. Developing the Web-app (30 minutes):

As you can see, the final results look really nice. This activation map is super informative for the radiologists to quickly pinpoint the area of infection on chest X-ray. To make our project become more user-frinedly. the final step is web-app development with an interactive UI. From our training the model, the best model was saved in a .pthf file extension. The trained weights and architecture from this .pth file are then deployed in a form of Django backend web app CovidScan.ai. While the minimal front-end of this web app is done using HTML, CSS, Jquery, Bootstrap. In our latter stage, the web-app will then be deployed and hosted on Debian server. 

## 5. Summary & Additional Resorces (5 minutes):
If you follow pace of time listed in this tutorial, in under 2 hours, you already explored a 5-step deep learning model building process using Pytorch. You also went over the concept of transfer learning and the architecture of our Resnet 152 model. In addition, you learnt to visualize the Activation Map using the last layer of our trained network. Eventually, you took a peek inside how this deep neural network is deployed to a web-app. 

The detailed web-developmeent process is not in the scope of this tutorial since we focus more on the Pytorch model to make the beginner user understand how we get to the finl visualization output from raw chest X-ray data. If you want to read more on how to implement the web-app, we can read the step-by-step instruction on this [gitlab tutorial](https://gitlab.com/sagban/pneumoscan-ppe).

For this project, we only implement a binary classification of 2 classes (COVID-19 and Normal). If you want to get more inspiration on building a AI-based product from scratch with multi-class data using Pytorch and FastAi, you can check out this other project created by our team called [HemoCount](https://devpost.com/software/hemonet-an-ai-based-white-blood-cell-count-platform?ref_content=user-portfolio&ref_feature=in_progress).

We hope you will have a good start by implementing this award-winning project, and be inspired to join other hackathons or datathon competition to build many other awesome AI products from scratch! Lastly, have a good hacking day, fellow hackers!


## References:
Crist, C. (2017, November 30). Radiologists want patients to get test results faster. Retrieved from https://www.reuters.com/article/us-radiology-results-timeliness/radiologists-want-patients-to-get-test-results-faster-idUSKBN1DH2R6 

Mauro Annarumma, Samuel J. Withey, Robert J. Bakewell, Emanuele Pesce, Vicky Goh, Giovanni Montana. (2019). Automated Triaging of Adult Chest Radiographs with Deep Artificial Neural Networks. Radiology; 180921 DOI: 10.1148/radiol.2018180921

 


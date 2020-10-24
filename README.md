# Building an AI COVID-19 Product from Scratch with Pytorch 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1H8SbSmrI3jjGL_8lWy1fPSeHsuL_Lp60)

Background:
In screening for Covid-19, patients can first be screened for flu-like symptoms using nasal swap to confirm their COVID-19 status. After 14 days of quarantine for confirmed cases, the hospital draws the patient’s blood and takes the patient’s chest X-ray. Chest X-ray is a golden standard for physicians and radiologists to check for the infection caused by the virus. An x-ray imaging will allow your doctor to see your lungs, heart and blood vessels to help determine if you have pneumonia. When interpreting the x-ray, the radiologist will look for white spots in the lungs (called infiltrates) that identify an infection. This exam, together with other vital signs such as temperature, or flu-like symptoms, will also help doctors determine whether a patient is infected with COVID-19 or other pneumonia-related diseases. The standard procedure of pneumonia diagnosis involves a radiologist reviewing chest x-ray images and send the result report to a patient’s primary care physician (PCP), who then will discuss the results with the patient.

 ![Alt text](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection/blob/main/Images/Fig1-Concept-idea.jpg)
 
 _Fig 1: Current chest X-ray diagnosis vs. novel process with PneumoScan.ai_

 ![Alt text](https://github.com/vicely07/Pneumonet-A-Pytorch-Chest-Xray-Pneumonia-Detection/blob/main/Images/Fig2-AI-vs-Manual.png)
 
_Fig 2: Chart of wait-time reduction of AI radiology tool (data from a simulation stud reported in Mauro et al., 2019). _

A survey by the University of Michigan shows that patients usually expect the result came back after 2-3 days a chest X-ray test for pneumonia. (Crist, 2017) However, the average wait time for the patients is 11 days (2 weeks). This long delay happens because radiologists usually need at least 20 minutes to review the X-ray while the number of images keeps stacking up after each operation day of the clinic. New research has found that an artificial intelligence (AI) radiology platform such as our CovidScan.ai can dramatically reduce the patient’s wait time significantly, cutting the average delay from 11 days to less than 3 days for abnormal radiographs with critical findings. (Mauro et al., 2019) With this wait-tine reduction, patients I critical cases will receive their results faster, and receive appropriate care sooner. In this blog post, we’ll build a machine learning pipeline to classify whether a patient has Pneumonia or not from chest x-ray images and then draw a heat-map  on areas that the model used to make these decisions. Below is a quick overview of the entire project.

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

# Pytorch Pneumonia localization

![Open In Colab](https://colab.research.google.com/drive/1GrPVEx9jKmnbrqrtnn2SrMiGC0iRMx66?usp=sharing)


 Not long ago, the machine learning group at Stanford University led by Adjunct Professor Andrew Ng released a paper on their computer vision algorithm which can detect and localize pneumonia from x-ray chest scans. Their algorithm outperformed the average practising radiologist at Stanford.Their algorithm achieved an accuracy of about 94%. Their project really inspired me so I decided to try it out. 

This project uses a pretrained resnet-152 neural network architecture to detect pneumonia in chest xray scans and  draw a heatmap to indicate areas the neural network used to classify it as an image containing pneumonia or not.
This project uses a pretrained resnet-152 neural network architecture to detect pneumonia in chest xray scans and  draw a heatmap to indicate areas the neural network used to classify it as an image containing pneumonia or not.
I used a training set consisting of almost 5300 images  of chest xray scans, a validation set of almost 430 images and a test set of 16 images. The algorithm achieved an accuracy of 92% on the training set, 81% percent on the validation set. There is a bit of high variance due to not having a large dataset.
Link to dataset: https://www.kaggle.com/nielspace/chest-x-ray/data 
If you do not want to go through the stress in training model from scratch, you can use an already trained model provided in the model folder.

# VIOLENCE AND WEAPONIZED VIOLENCE DETECTION

# INTRODUCTION
This repository introduces Salient Image, a method for Violence and Weaponized Violence Detection from Smart Surveillance Systems (CCTV). The following algorithms were employed for training and inference on the 5 by 3 and 3 by 2 versions of the Salient Image:
  - VGG16
  - VGG19
  - ResNet50
  - ResNet101
  - DenseNet121
  - EfficientNetB0
  - InceptionV3
  
  1. For the paper, download from [here](https://arxiv.org/abs/2207.12850)
  2. For the dataset, download from [here](www.kaggle.com/dataset/75806dc0d1bc0fccd0cedaf117979ffa2f2ae5c3c7af3cdd78b9f4cc14d96013)
  
  Also, if you use this repository or dataset, please make sure to cite our [paper](https://arxiv.org/abs/2207.12850). Thank you.
  
# RESULTS
Our results for the 3 x 2 and 5 x 3 salient arrangements are as follows:

| Model Architecture | 3 x 2 Accuracy (%) | 5 x 3 Accuracy (%) |
|--------------------|---------------|:-------------:|
| VGG16              |     98.14     |         97.98 |
| VGG19              |     98.31     |         96.42 |
| ResNet50           |     96.79     |         96.51 |
| ResNet101          |     97.89     |         91.91 |
| DenseNet121        |     98.65     |         96.60 |
| EfficientNetB0     |     84.04     |         82.26 |
| InceptionV3        |     98.65     |         94.12 |

## GRAD-CAM Results

**Using Grad-CAM for explaining how 5 different models made their inferences on the 3 x 2 and 5 x 3 salient arrangements, we got:**

**In order: DenseNet121, EfficientNetB0, InceptionV3, ResNet50, VGG16**

**3 x 2**

<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/densenet121_3x2.jpg" width="150">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/efficientnet_3x2.jpg" width="150">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/inception_3x2.jpg" width="150">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/resnet50_3x2.jpg" width="150">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/vgg16_3x2.jpg" width="150">

**5 x 3**

<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/densenet121_5x3.jpg" width="180">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/efficientnet_5x3.jpg" width="180">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/inception_5x3.jpg" width="180">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/resnet50_5x3.jpg" width="180">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/vgg16_5x3.jpg" width="180">


## VIDEO EVALUATION

**Evaluating our trained models on a CCTV video using VGG16_3x2, Inception_3x2, DenseNet_5x3:**


<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/VGG16_3_2_output.gif" width="500">

<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/Inception_3_2_output.gif" width="500">

<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/Densenet_5_3_output%20.gif" width="500">

# USAGE
## ENVIRONMENT SETUP
Conda create and install packages needed by the environment. Use this code:
```
conda env create -f env.yml
```

## PREPROCESSING
1. Use the ```dataUtils.save_to_frame``` to extract frames from the video dataset. If you want to use the SCVD dataset, download from here
2. Use the ```dataUtils.merge_frames``` to apply your selected configuration of the SALIENT IMAGE

## TRAINING
In the ```models``` folder, there are different notebooks using different DCNN architectures. Choose whichever fits your purpose, or create a new one and follow the models examples.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-smart-city-security-violence-and/violence-and-weaponized-violence-detection-on)](https://paperswithcode.com/sota/violence-and-weaponized-violence-detection-on?p=towards-smart-city-security-violence-and)

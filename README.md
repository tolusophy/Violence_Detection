# SSIVD-Net: A Novel Salient Super Image Classification \& Detection Technique for Weaponized Violence

<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/VGG16_3_2_output.gif" width="270" height="270"> <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/Inception_3_2_output.gif" width="270" height="270"> <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/Densenet_5_3_output%20.gif" width="270" height="270">

# INTRODUCTION

Our project focuses on the detection of violence and weaponized violence in CCTV footage using a comprehensive approach. We have introduced the Smart-City CCTV Violence Detection (SCVD) dataset, specifically designed to facilitate the learning of weapon distribution in surveillance videos. To address the complexities of analyzing 3D surveillance video, we propose a novel technique called SSIVD-Net (Salient-Super-Image for Violence Detection). Our method reduces data complexity and dimensionality while improving inference, performance, and explainability through the use of Salient-Super-Image representations. We also introduce the Salient-Classifier, a novel architecture that combines a kernelized approach with a residual learning strategy. Our approach outperforms state-of-the-art models in detecting both weaponized and non-weaponized violence instances. By advancing violence detection and contributing to the understanding of weapon distribution, our research enables smarter and more secure cities while enhancing public safety measures.

<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/sivi.png"> 
<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/join.png"> 

  
  1. For the updated paper, link coming soon
  2. For the dataset, download from [here](https://www.kaggle.com/datasets/75806dc0d1bc0fccd0cedaf117979ffa2f2ae5c3c7af3cdd78b9f4cc14d96013)
  
  Also, if you use this repository or dataset, please make sure to cite our [paper](https://arxiv.org/abs/2207.12850). Thank you.
  
    @misc{scvd-salient,
          doi = {10.48550/ARXIV.2207.12850},
          url = {https://arxiv.org/abs/2207.12850},
          author = {Aremu, Toluwani and Zhiyuan, Li and Alameeri, Reem},
          title = {Any Object is a Potential Weapon! Weaponized Violence Detection using Salient Image },
          publisher = {arXiv},
          year = {2022},
          copyright = {Creative Commons Attribution 4.0 International}
    }
  
# RESULTS


## Other Ablations
The following algorithms were originally employed for training and inference on the 5 by 3 and 3 by 2 versions of our datacentric approach:
  - VGG16
  - VGG19
  - ResNet50
  - ResNet101
  - DenseNet121
  - EfficientNetB0
  - InceptionV3

**Using Grad-CAM for explaining how 5 different models made their inferences on the 3 x 2 and 5 x 3 salient arrangements, we got:**

**In order: DenseNet121, EfficientNetB0, InceptionV3, ResNet50, VGG16**

**3 x 2**

<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/densenet121_3x2.jpg" width="161">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/efficientnet_3x2.jpg" width="161">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/inception_3x2.jpg" width="161">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/resnet50_3x2.jpg" width="161">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/vgg16_3x2.jpg" width="161">

**5 x 3**

<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/densenet121_5x3.jpg" width="161">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/efficientnet_5x3.jpg" width="161">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/inception_5x3.jpg" width="161">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/resnet50_5x3.jpg" width="161">  <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Grad-CAM%20output/vgg16_5x3.jpg" width="161">


## VIDEO EVALUATION

**Evaluating our trained models on a CCTV video using VGG16_3x2, Inception_3x2, DenseNet_5x3:**




# USAGE
## ENVIRONMENT SETUP
Conda create and install packages needed by the environment. Use this code:
```
conda env create -f env.yml
```

## PREPROCESSING
1. Use the ```dataUtils.save_to_frame``` to extract frames from the video dataset. If you want to use the SCVD dataset, download from [here](www.kaggle.com/dataset/75806dc0d1bc0fccd0cedaf117979ffa2f2ae5c3c7af3cdd78b9f4cc14d96013)
2. Use the ```dataUtils.merge_frames``` to apply your selected configuration of the SALIENT IMAGE

## TRAINING
In the ```models``` folder, there are different notebooks using different DCNN architectures. Choose whichever fits your purpose, or create a new one and follow the models examples.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-smart-city-security-violence-and/violence-and-weaponized-violence-detection-on)](https://paperswithcode.com/sota/violence-and-weaponized-violence-detection-on?p=towards-smart-city-security-violence-and)

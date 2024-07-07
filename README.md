# SSIVD-Net: A Novel Salient Super Image Classification \& Detection Technique for Weaponized Violence

<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/VGG16_3_2_output.gif" width="270" height="270"> <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/Inception_3_2_output.gif" width="270" height="270"> <img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/Densenet_5_3_output%20.gif" width="270" height="270">

# INTRODUCTION
Our project focuses on the detection of violence and weaponized violence in CCTV footage using a comprehensive approach. We have introduced the Smart-City CCTV Violence Detection (SCVD) dataset, specifically designed to facilitate the learning of weapon distribution in surveillance videos. To address the complexities of analyzing 3D surveillance video, we propose a novel technique called SSIVD-Net (Salient-Super-Image for Violence Detection). Our method reduces data complexity and dimensionality while improving inference, performance, and explainability through the use of Salient-Super-Image representations. We also introduce the Salient-Classifier, a novel architecture that combines a kernelized approach with a residual learning strategy. Our approach outperforms state-of-the-art models in detecting both weaponized and non-weaponized violence instances. By advancing violence detection and contributing to the understanding of weapon distribution, our research enables smarter and more secure cities while enhancing public safety measures.

# METHOD
<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/sivi.png"> 
<img src="https://github.com/Ti-Oluwanimi/Violence_Detection_Main/blob/main/Result/join.png"> 

Below is a table that shows the layer arrangements of Salient-Classifier architectures and their number of parameters:
| Classifier | Layer Arrangement | Minimal Block(m) | Basic Block(b) | Bottle Neck(n) |
|------------|------------------|------------------|----------------|----------------|
| SaliNet-2  | 1, 1, 0, 0       | 1.8              | 4.9            | 8.0            |
| SaliNet-4  | 1, 1, 1, 1       | 1.8              | 4.9            | 8.0            |
| SaliNet-8  | 2, 2, 2, 2       | 4.9              | 11.2           | 14.0           |
| SaliNet-16 | 3, 4, 6, 3       | 10.0             | 21.3           | 23.5           |

# RESULTS
## Main Results
**Eliminating parameters using the Salinet-2m variant:**
| k - grid_shape | Sampler      | Aspect Ratio | Accuracy(%) | AP(%) | Inference time (s) |
|----------------|--------------|--------------|-------------|-------|--------------------|
| 4 - 2x2        | uniform      | square       | 78.4        | 80.5  | 0.04               |
| 4 - 2x2        | random       | square       | 75.5        | 79.9  | 0.05               |
| 4 - 2x2        | continuous   | square       | 74.4        | 76.2  | 0.04               |
| 4 - 2x2        | mean_abs     | square       | 71.1        | 77.7  | 0.15               |
| 4 - 2x2        | LK           | square       | 69.6        | 78.2  | 0.21               |
| 4 - 2x2        | centered     | square       | 73.2        | 78.7  | 0.04               |
| 4 - 2x2        | consecutive  | square       | 70.4        | 79.4  | 0.04               |
| 6 - 3x2        | uniform      | 144p_A       | 78.9        | 81.2  | 0.05               |
| 6 - 3x2        | uniform      | 144p_B       | 79.7        | 81.9  | 0.05               |
| 6 - 3x2        | uniform      | 240p_A       | 80.9        | 84.0  | 0.05               |
| 6 - 3x2        | uniform      | 240p_B       | 81.3        | 84.2  | 0.05               |
| 6 - 3x2        | uniform      | 360p_A       | 78.4        | 81.9  | 0.05               |
| 6 - 3x2        | uniform      | 360p_B       | 82.4        | 83.8  | 0.05               |
| 6 - 3x2        | uniform      | 480p_A       | 83.0        | 83.4  | 0.05               |
| 6 - 3x2        | uniform      | 480p_B       | 83.0        | 83.4  | 0.06               |
| 9 - 3x3        | uniform      | square       | 84.7        | 85.0  | 0.06               |
| 12 - 4x3       | uniform      | 480p_A       | 86.6        | 89.6  | 0.07               |
| 15 - 5x3       | uniform      | 480p_A       | 84.3        | 86.8  | 0.08               |

**Comparing our Salient-Classifers with SOTA:**
| Model                           | Num_Params (M) | Accuracy (%) |
|---------------------------------|----------------|--------------|
| FGN                             | 0.3            | 74.4         |
| Conv-LSTM                       | 47.4           | 71.6         |
| Sep-Conv-LSTM                   | 0.4            | 78.4         |
| SaliNet-2m                      | 1.8            | 86.6         |
| SaliNet-4m                      | 1.8            | 83.1         |
| SaliNet-8m                      | 4.9            | 77.8         |
| SaliNet-2b                      | 4.9            | 75.9         |
| SaliNet-2n                      | 8.0            | 78.8         |

**Comparing our Salient-Classifiers with SOTA on other datasets:**
| Method              | Model          | MovieFight | HockeyFight | SCVD |
|---------------------|----------------|------------|-------------|------|
|                     | C3D            | 100.0      | 96.5        | 82.8 |
| 3D-CNNs             | I3D            | 100.0      | 98.5        | 85.8 |
|                     | FGN            | 100.0      | 98.0        | 87.3 |
|                     | Conv-LSTM      | 100.0      | 97.1        | 77.0 |
| Conv-LSTM           | Bi-Conv-LSTM   | 100.0      | 98.1        | -    |
|                     | Sep-Conv-LSTM  | 100.0      | 99.5        | 89.3 |
|                     | SaliNet-2m     | 100.0      | 100.0       | 88.5 |
| Salient-Classifiers | SaliNet-2b     | 100.0      | 100.0       | 89.7 |
|                     | SaliNet-2n     | 100.0      | 100.0       | 90.3 |

# USAGE
## ENVIRONMENT SETUP
Libraries:
- Pytorch
- Numpy
- OpenCV
- tqdm

## TRAINING
1. In the ```main.py``` file, edit the parameters to match the task you would use it for.
2. Ensure that the video dataset are arranged accordingly, just like the structure below.
   - VideoDataset
      - Train
        - Class A
        - Class B
      - Test
        - Class A
        - Class B
3. Go to the ```Scripts/ssi.py``` file, and edit the class names.
4. run ```python main.py```

## NOTE

1. For the updated paper, [link](https://www.researchsquare.com/article/rs-3024402/v1)
2. For the dataset, download from [here](https://www.kaggle.com/datasets/75806dc0d1bc0fccd0cedaf117979ffa2f2ae5c3c7af3cdd78b9f4cc14d96013). A preprocessed version can be downloaded [here](https://drive.google.com/file/d/16Uk5AAWo6UorGyQ_YaUUn6CWxPxMHE0K/view?usp=sharing). If you use our dataset or code, please cite our paper and like our repository.
```
BIB: @InProceedings{
         10.1007/978-3-031-62269-4_2,
         author="Aremu, Toluwani
         and Zhiyuan, Li
         and Alameeri, Reem
         and Khan, Mustaqeem
         and Saddik, Abdulmotaleb El",
         editor="Arai, Kohei",
         title="SSIVD-Net: A Novel Salient Super Image Classification and Detection Technique for Weaponized Violence",
         booktitle="Intelligent Computing",
         year="2024",
         publisher="Springer Nature Switzerland",
         address="Cham",
         pages="16--35",
         isbn="978-3-031-62269-4"
}

Springer Nature: Aremu, T., Zhiyuan, L., Alameeri, R., Khan, M., Saddik, A.E. (2024). SSIVD-Net: A Novel Salient Super
Image Classification and Detection Technique for Weaponized Violence. In: Arai, K. (eds) Intelligent Computing. SAI 2024.
Lecture Notes in Networks and Systems, vol 1018. Springer, Cham. https://doi.org/10.1007/978-3-031-62269-4_2.

APA: Aremu, T., Zhiyuan, L., Alameeri, R., Khan, M., & Saddik, A. E. (2024, June). SSIVD-Net: A Novel Salient Super Image
Classification and Detection Technique for Weaponized Violence. In Science and Information Conference (pp. 16-35).
Cham: Springer Nature Switzerland.
```

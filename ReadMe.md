# NAML Practical Project AA 2023-24
### Authors
- Gabriele Carrino
- Giacomo Brunetta

## Goal of the project: 
This project aims to reproduce, and possibly improve, the results obtained by Dhevan S. Lau and Ritesh Ajoodha in the paper: "Music Genre Classification A Comparative Study between Deep Learning and Traditional Machine  Learning Approaches".
The original research paper compared five off-the-shelf machine learning classifiers with two deep learning models on the task of music genre classification on the popular GTZAN dataset. The hypothesis presented was that a deep-learning approach would have outperformed the traditional models. Still, the results obtained were lacking so the authors ended up neither rejecting nor accepting the hypothesis.


## Repository Structure
The project is organized in two sections.
- The *Report* folder contains the original paper and the report in which we extensively comment on the results we obtained by reproducing it.
- The *Code* folder contains:
  - the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (without the audio files)
  - 5 Jupiter Notebooks that we used to reproduce the results
  - the trained PyTorch models
  - some utils

## Methodology and Requirements
All the results are reproduced using Python. The machine learning models and the dataset preprocessing are implemented with Scikit-learn, while the deep learning models are made with Pythorch.

### Machine Learning Models
| Model               | Description                                                                                      |
|---------------------|--------------------------------------------------------------------------------------------------|
| Logistic Regression | Multinomial classification, LBFGS solver, Ridge regularization, 400 iterations.                  |
| K-nearest Neighbor  | 1-Nearest Neighbor algorithm.                                                                   |
| Support Vector Machine | One-vs-One strategy for multiclass classification.                                               |
| Random Forest       | 1000 decision trees, maximum depth of 10.                                                        |
| Multilayer Perceptron | Input layer: 57, Hidden layer 1: 5000, Hidden layer 2: 10, Output layer: 10. LBFGS solver, Relu activation, alpha=10e-5. |

### Deep Learning Models
| Model                      | Description                                                                                                                                                  |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Deep Neural Network        | Input: 57 neurons, 4 hidden layers: (512, 256, 128, 64), Output: 10 neurons. ReLU activation. Dropout: 0.2 between each hidden layer, 0.5 before output. AdamW optimizer, weight decay: 1e-5. |
| Convolutional Neural Network | 5 convolutional blocks: Convolution (3x3 filter, 1x1 stride, mirrored padding), ReLU activation, Max pooling (2x2 window, 2x2 stride), Dropout: 0.2. Filter sizes: (16, 32, 64, 128, 256). 1 Global Average Pooling layer. AdamW optimizer, weight decay: 1e-5. |

## Conclusions
Dhevan S. Lau and Ritesh Ajoodha decided to neither reject nor accept the hypothesis presented, we on the other end have found sufficient evidence to accept the hypothesis.
Both the Deep and Convolutional Neural networks outperformed by a significative margin the classical machine learning techniques on the 30s dataset while performing on par on the 3s dataset.

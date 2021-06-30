# Emotion Recognition
## Team
Xinyi Zhao (zxy1307@bu.edu)\
Shihao Xing (ericxing@bu.edu)
## Slides
https://drive.google.com/file/d/1Wyub-xqXgsnzTm_MuaGo8FN6kOuCTE2I/view?usp=sharing
## Motivation
1. To reduce the development of fatigue driving and other situations by identifying the status of the driver.
2. To improve players’ human-computer interaction experience with emotion recognition.
## Goal
1. Design a system that can predict and recognize the classification of facial emotions into one of the seven facial emotion categories based on feature extraction using the Convolution Neural Network algorithm with TensorFlow and Keras
2. This approach enables to classify seven basic emotions consist of angry, disgust, fear, happy, neutral, sad and surprise from image data.
## Neural Network Architecture
Our designed neural network uses EfficientNetB1 as base mode, and adding one FC layer.\
![Image text](https://github.com/EricXSH/CS523-Project/blob/main/img_files/EfficientNet%20B1.png)
## DataSet
[FER2013](https://www.kaggle.com/msambare/fer2013/download)
## Prerequisites
If use conda, use the following commands to install the above packages:\
•	conda install tensorflow\
•	conda install matplotlib\
•	conda install numpy \
•	conda install -c anaconda scikit-learn\
•	conda install pydot\
•	conda install graphviz

If use pip, use use the following commands to install the above packages:\
•	pip install tensorflow\
•	pip install matplotlib \
•	pip install numpy \
•	pip install sklearn\
•	pip install pydot\
•	pip install graphviz

## Runtime Environment
Python 3.8 \
Tensorflow 2.5
## How Compile the Code
1. Download the [FER2013](https://www.kaggle.com/msambare/fer2013/download) dataset.
2. Unzip the file and rename the folder as 'fer2013'.
3. Download [EfficientNetB1.py](https://github.com/EricXSH/CS523-Project/blob/main/EfficientNetB1.py) and [Demo.ipynb](https://github.com/EricXSH/CS523-Project/blob/main/Demo.ipynb).
4. Before compiling EfficientNetB1.py, make sure your GPU is enabled for neural network training.
5. Compile 'Designed NN.py'. When the training completes, the best trained model will be saved to 'EfficientNetB1.h5' and the graph of training and validation loss with each epoch will be demonstrated.
6. Compile 'EfficientNetB1.py'. When the training completes, the best trained model will be save to 'EfficientNetB1_V1.h5'.
7. Compile 'Demo.ipynb'. The training and test accuracy of both designed neural network and pure EfficientNetB1 and the designed neural network confusion matrix of test dataset will be shown.
## Training Result
### Loss and Accuracy Graph of Designed Neural Network
![Image text](https://github.com/EricXSH/CS523-Project/blob/main/img_files/EffNet%20T%26T%20accuracy.png)
### Result of pure EfficientNetB1
![Image text](https://github.com/EricXSH/CS523-Project/blob/main/img_files/Pure%20EfficientNetB1%20Result.png)
### Result of Designed Neural Network
![Image text](https://github.com/EricXSH/CS523-Project/blob/main/img_files/Designed%20NN%20result.png)
## Confusion Matrix of Test Dataset
![Image text](https://github.com/EricXSH/CS523-Project/blob/main/img_files/Confusion%20Matrix%20of%20Test%20Set.png)
## Reference
1. Tan, M. and Le, Q. V. Tan, Mingxing, and Quoc V. Le. "Efficientnet: Rethinking Model Scaling For Convolutional Neural Networks". Arxiv.Org, 2019, https://arxiv.org/abs/1905.11946. Accessed 29 June 2021.
2. https://www.kaggle.com/msambare/fer2013/data
3. https://www.kaggle.com/aroraumang/facial-emotion-recogination-82-accuracy

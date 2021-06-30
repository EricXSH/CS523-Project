# Emotion Recognition
## Motivation
1. To reduce the development of fatigue driving and other situations by identifying the status of the driver.
2. To improve players’ human-computer interaction experience with emotion recognition.
## Goal
1. Design a system that can predict and recognize the classification of facial emotions into one of the seven facial emotion categories based on feature extraction using the Convolution Neural Network algorithm with TensorFlow and Keras
2. This approach enables to classify seven basic emotions consist of angry, disgust, fear, happy, neutral, sad and surprise from image data.
## DataSet
[FER2013](https://www.kaggle.com/msambare/fer2013/download)
## Runtime Environment
Python 3.8 \
Tensorflow 2.3
## How Compile the Code
1. Download the [FER2013](https://www.kaggle.com/msambare/fer2013/download) dataset.
2. Unzip the file and rename the folder as 'fer2013'.
3. Download [EfficientNetB1.py](https://github.com/EricXSH/CS523-Project/blob/main/EfficientNetB1.py) and [Demo.ipynb](https://github.com/EricXSH/CS523-Project/blob/main/Demo.ipynb).
4. Before compile EfficientNetB1.py, make sure your GPU is enabled for neural network training.
5. When the training completes, the best trained model will be saved to 'EfficientNetB1.h5' and the graph of training and validation loss with each epoch will be demonstrated.
6. 



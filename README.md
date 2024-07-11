# LipReadingNeuralNetwork
LipNet paper implementation using Tensorflow

This repository contains the code and resources for reproducing the LipNet model based on the LipNet paper [here](https://arxiv.org/pdf/1611.01599). The model is built using TensorFlow and focuses on lip reading at the character level.

## Model Overview

The LipNet model operates at the character level and uses the following components:

- **Spatiotemporal Convolutional Neural Networks (STCNNs)**
- **Recurrent Neural Networks (RNNs)**
- **Connectionist Temporal Classification (CTC) Loss**

### Key Features

- **CTC Loss Function**: The Connectionist Temporal Classification (CTC) loss calculates the loss between a continuous (unsegmented) time series and a target sequence. Unlike the standard cross-entropy function, CTC avoids repetition of letters.
- **Bidirectional LSTM**: Used to account for temporal changes in letter recognition, allowing the model to process the sequence both forwards and backwards.

## Dataset

The dataset used is the **Grid Corpus (Pre-Processed)**. It is pre-processed specifically for training LipNet models. The preprocessing steps include:

1. Scaling down the video frame to focus on the mouth area.
2. Converting the videos to greyscale.
3. Converting the greyscaled videos to tensors.
4. Converting the tensors to GIFs for training.

## Streamlit App

A Streamlit app is included to provide a user-friendly interface for demonstrating how the model works. All the necessary files for running the app are provided in this repository.

## Images

![Screenshot 2024-07-11 114510](https://github.com/aaranayaadi/LipReadingNeuralNetwork/assets/37096846/bb21ba1a-84db-45cd-969e-c396a924e675)
![Screenshot 2024-07-11 114523](https://github.com/aaranayaadi/LipReadingNeuralNetwork/assets/37096846/9ab63645-3fc8-42cd-a68b-d24531f1f79e)
![Screenshot 2024-07-11 114537](https://github.com/aaranayaadi/LipReadingNeuralNetwork/assets/37096846/156c80c4-660a-4717-8c85-1f141a5734bc)
![Screenshot 2024-07-11 114602](https://github.com/aaranayaadi/LipReadingNeuralNetwork/assets/37096846/f4fc6900-081a-4f5d-98a4-fed3a7c3d83e)


## Running the Model

To run and test the model, follow these steps:

1. **Import Libraries**: Ensure all necessary libraries are imported.
2. **Download Dataset**: Run the first `gdown` cell to download and unzip the dataset.
3. **Run Model Compilation**: Execute all the cells up to the model compilation step.
4. **Use Pre-trained Weights**: No need to train the model again. I have provided the trained weights (trained till the 96th epoch) via Google Drive. Run the next `gdown` cell to import and use these weights.
5. **Test the Model**: Test the model on videos from the dataset by running the subsequent cells.

## References
- [LipNet Paper](https://arxiv.org/pdf/1611.01599)



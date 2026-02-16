# Teath Disease Classification using CNN

## 📖 Overview
This project builds a Convolutional Neural Network (CNN) model to classify skin diseases into 7 categories using deep learning techniques.

The goal is to assist in early detection and automated diagnosis support.

## 🚀 Features
- Custom CNN architecture
- Data augmentation
- Model evaluation with confusion matrix
- Streamlit web app for predictions
- Dockerized deployment

## 📊 Dataset
- 7 Teeth disease classes
- Images resized to 256x256
- Balanced using augmentation

## 🏗 Model Architecture
- Input: 256x256x3
- Convolutional layers + BatchNorm
- GlobalAveragePooling
- Dense layer with Softmax

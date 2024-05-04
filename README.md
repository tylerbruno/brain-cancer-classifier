# Brain Cancer MRI Classifier

This project is a machine learning classifier designed to differentiate between four categories of brain conditions: glioma, meningioma, no tumor, and pituitary. The classifier is built using PyTorch and utilizes pandas, numpy, and matplotlib for data processing, manipulation, and visualization. The goal of this project is to provide a robust tool for medical imaging analysis and to contribute to the early diagnosis of brain cancer. This project will built as a tool to learn more about ML/Deep Learning with the help of documentation, videos, Google, articles, and more. I was unable to find the specific pieces of information I used, but I wanted to properly cite that I used online resources here.

## Project Overview
### Description
Brain cancer is a critical health issue that requires timely diagnosis for effective treatment. This project focuses on building a deep learning classifier to analyze MRI images and classify them into one of four categories: Glioma, Meningioma, No Tumor, and Pituitary. 

The project leverages a dataset of MRI images to train a neural network using PyTorch. The classifier uses a convolutional neural network (CNN) architecture to extract features from the images and make predictions. The dataset can be found on Kaggle at the following link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset. The training and evaluation processes are supported by pandas and numpy for data handling, while matplotlib is used for visualizations and analysis.

### Project Structure
model.py: Contains the Brain Cancer Classifier Model. \
README.md: This file. 

### How to Run Locally
Originally, this project was on Kaggle to have access to the dataset and GPU, and transferred here for submission. To run locally, installing the following necessary dependencies will be required: Python 3.7 or above, PyTorch, pandas, numpy, and matplotlib. The dataset can be accessed directly using the link above or already in the code. Running the model.py file will train, evaluate, and visualize the results. 

### Screenshots
![image](https://github.com/tylerbruno/braincancerclassifier/assets/125921211/541c51cf-005c-4e96-b7ca-c64b3d0bad0e) \
The types of the images we are working with are shown above. 

The training/evaluation can be seen here: 
![image](https://github.com/tylerbruno/braincancerclassifier/assets/125921211/4668b589-2361-4f2c-a29e-3a5dd2927fa1)

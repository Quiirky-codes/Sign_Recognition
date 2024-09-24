# Project Overview

This project aims to eradicate the communication barrier between Deaf and hearing individuals by developing a system to detect and interpret American Sign Language (ASL) gestures. 
Initially, we use a Random Forest classifier for gesture recognition. In the next phase, we integrate deep learning and computer vision techniques to improve accuracy and real-time performance.

# Objectives
* ***Improve Communication Accessibility:*** Facilitate real-time ASL translation to text or speech.
* ***Enhance Educational Tools:*** Develop applications for learning and practicing ASL.
* ***Promote Social Integration:*** Improve mutual understanding between Deaf and hearing communities.
* ***Support Accessibility in Public Services:*** Integrate ASL recognition into public service platforms.
* ***Encourage Technological Innovation:*** Explore advancements in AI-driven accessibility solutions.

# Architecture

## Phase 1: Random Forest Classifier

* ***Data Acquisition and Preprocessing:*** Video feed or image frames are preprocessed by resizing, normalizing, and augmenting.
* ***Feature Extraction:*** Use OpenCV for real-time video capture, hand detection, and feature extraction.
* ***Random Forest Classifier:*** Train a Random Forest classifier with the extracted features.
* ***Evaluation:*** Assess the model using accuracy, precision, recall, and F1-score.
* ***Integration:*** Implement real-time gesture recognition and integrate with user interfaces.

## Phase 2: Deep Learning and Computer Vision

* ***Model Selection:*** Utilize CNNs or transfer learning with pre-trained models for feature extraction.
* ***Image Preprocessing:*** Advanced preprocessing steps, including normalization and augmentation.
* ***Model Architecture:*** Convolutional layers for feature extraction followed by fully connected layers for classification.
* ***Training:*** Use optimizers like Adam and loss functions like categorical cross-entropy.
* ***Evaluation:*** Validate the model with performance metrics and refine through hyperparameter tuning.
* ***Real-Time Application:*** Implement and test the model in real-time environments.

# Installation
* To set up the project locally:
  
```
git clone https://github.com/Quiirky-codes/Sign_To_Text_Recognition.git
```

```
cd Sign_To_Text_Recognition
```

* Create a virtual environment:
  
```
python -m venv venv
```

```
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

* Install the required dependencies:

```
pip install -r requirements.txt
```

# Usage

* ***Data Preprocessing:*** Run the script to preprocess and augment data.

```
python create_imgs.py
```

```
python create_dataset.py
```

***Running this should create a pickle file***

* Training the model

```
python train_classifier.py
```

***Running this should create a model file***

```
python inference_classifier.py
```

# Data

The dataset consists of images or video frames of ASL gestures. Ensure you have the appropriate dataset downloaded and placed in the data/ directory.

# Contributors

* [Amith M Jain](https://github.com/Quiirky-codes)
* [Dheeraj N](https://github.com/Dheeraj44444)

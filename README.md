# Fibromyalgia Prediction

This repository contains a Python script for predicting the probability of a patient having fibromyalgia using a Random Forest Classifier. The script takes various patient characteristics as input and uses a pre-trained model to make predictions.

## Getting Started

Follow these instructions to set up and run the code on your local machine.

### Prerequisites

Make sure you have the following Python packages installed:

- pandas
- scikit-learn


Follow the prompts to enter patient information, and the script will predict the probability of fibromyalgia for the given patient.

## Dataset

The dataset used for this project is stored in the `Fibromyalgia_patients.csv` file. It contains the following columns:

- `Patient ID`: Unique identifier for each patient.
- `Age`: Age of the patient.
- `Gender`: Gender of the patient (0 for male, 1 for female).
- `Family History`: Family history of fibromyalgia (0 for no, 1 for yes).
- `Emotional Stress`: Presence of emotional stress (0 or 1).
- `Physical Trauma`: History of physical trauma (0 or 1).
- `Previous Infection`: Previous infection (0 or 1).
- `Depression`: Presence of depression (0 or 1).
- `Rheumatoid Arthritis`: Rheumatoid arthritis diagnosis (0 or 1).
- `Chronic Fatigue Syndrome`: Chronic fatigue syndrome diagnosis (0 or 1).
- `Arthritis`: Arthritis diagnosis (0 or 1).
- `Migraine`: Migraine diagnosis (0 or 1).
- `Generalized Pain`: Generalized pain (0 or 1).
- `Anxiety`: Presence of anxiety (0 or 1).
- `Fatigue`: Presence of fatigue (0 or 1).
- `Fibromyalgia`: The target variable indicating whether the patient has fibromyalgia (0 for no, 1 for yes).

## Model

The script uses a Random Forest Classifier to make predictions. It first loads a pre-trained model and then prompts the user for patient information to predict the probability of fibromyalgia.

## Results

After inputting patient information, the script will display the probability of the patient having fibromyalgia.

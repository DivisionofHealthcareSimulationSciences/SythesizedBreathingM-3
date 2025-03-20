# SynthesizedBreathingN^3

## Overview:
This project aims to develop a system that generates breathing sounds for medical simulation manikins. The generated sounds can be adjusted based on variables such as demographic factors (e.g., age, gender) and health conditions (e.g., asthma, COPD, pneumonia), to help train students in diagnosing medical conditions accurately. The system is designed to receive input about the patient's condition and the location of the stethoscope, and then generate the corresponding breath sounds to be transmitted through a Bluetooth-enabled stethoscope speaker system.

## Structure:
This repository contains the generative models we are developing to achieve the task of simulating breathing sounds based on different variables.

The datasets were restructured to categorize based on the following features: 
1. Age 
2. Sex
3. Diagnosis 
4. Anterior/Posterior position 
5. Lateral left/right position 

The respiratory conditions represented across both datasets include:

- Healthy
- Asthma
- Bronchiectasis
- Bronchiolitis
- COPD (Chronic Obstructive Pulmonary Disease)
- Heart Failure
- Lower Respiratory Tract Infection
- Lung Fibrosis
- Pleural Effusion
- Pneumonia
- Upper Respiratory Tract Infections

## Details:
The datasets used for this project can be found on Kaggle at the following links:

- [Dataset 1](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)
- [Dataset 2](https://www.kaggle.com/datasets/arashnic/lung-dataset/data)

### Dataset 1:
Dataset 1 includes information on various patient attributes such as age, sex, respiratory condition, adult BMI (kg/m^2), child weight (kg), and child height (cm). The dataset filenames are structured with five elements:
1. Patient number
2. Recording index 
3. Chest location (e.g., Trachea, Anterior left, Anterior right, Posterior left, Posterior right, Lateral left, Lateral right)
4. Acquisition mode (either sequential/single-channel or simultaneous/multi-channel)
5. Type of stethoscope used (AKG C417L, 3M Littmann Classic, 3M Electronic, or WelchAllyn Meditron Master Elite)


### Dataset 2: 
Dataset 2 also includes information on various patient attributes such as age, sex, and respiratory condition. The dataset filenames are structured with four elements:
1. Age 
2. Sex
3. Location (Posterior/Anterior, Left/Right, Upper/Middle/Lower)
4. Breath sounds (Inspiratory/Experiatory, Wheezes, Crackles, Normal, crepitations)


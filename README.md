# Credit_Risk_Analysis
Module 17 ~ Machine Learning


## Resources
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
CSV: LoanStats_2019Q1.csv
Jupyter Notebook


## Project Overview & Challenge
The purpose of this module is to apply machinine learning different techniques to train and evaluate models with unbalanced classes. Using scikit-learn and imbalanced-learn to train and evaluate models to determine credit card risk using a credit card credit dataset from LendingClub, a peer-to-peer lending services company. Six different techniques were used to train and evaluate models with unbalanced classes to determine credit risk: Oversampling with the RandomOverSampler and SMOTE algorithms, Undersampling with the ClusterCentroids algorithm, a combination of over and under sampling with the SMOTEENN algorithm, and two machine learning models that reduce bias - the BalancedRandomForestClassifer and EasyEnsembleClassifier. The following is an analysis of the models and their results. 

## Results:

## Oversampling 
Two oversampling algorithms to determine which algorithm results in the best performance. Oversample the data using the naive random oversampling algorithm and the SMOTE algorithm

### Naive Random Oversampling
-   The balanced accuracy score for this model is around 66%. This is a fairly positive score, but not excellent.
-   The precision scores for this model at 1% 
-   The recall score for this model is at 70%

![Oversampling_NaiveRandom](https://user-images.githubusercontent.com/80075982/125714968-3312dbc0-84c4-4dde-a479-43bc77594a8c.png)

### SMOTE Oversampling
-   The balanced accuracy score for this model is around 66%. This is a fairly positive score, but not excellent.
-   The precision scores for this model at 1% 
-   The recall score for this model is at 63%

![Oversampling_SMOTE](https://user-images.githubusercontent.com/80075982/125714970-b4b58033-ec42-43aa-b617-c8f338acd6a6.png)

## Undersampling
Undersampling algorithms to determine which algorithm results in the best performance compared to oversampling algorithms.

-   The balanced accuracy score for this model is around 66%. This is a fairly positive score, but not excellent.
-   The precision scores for this model at 1% 
-   The recall score for this model is at 72%

![Undersampling](https://user-images.githubusercontent.com/80075982/125714961-89ec9e07-4cb0-4ca6-a9c7-3cada25333c4.png)

## Combination (over and Under) Sampling
To test a combination over and under sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms.

-   The balanced accuracy score for this model is around 66%. This is a fairly positive score, but not excellent.
-   The precision scores for this model at 1% 
-   The recall score for this model is at 72%

![Combination_Sampling](https://user-images.githubusercontent.com/80075982/125714964-f7fe0be4-ca53-4a67-acef-b59e88400ad8.png)

## Ensemble Learners
Two ensemble algorithms to determine which algorithm results in the best performance. Train a Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier.

### Balanced Random Forest Classifier
-   The balanced accuracy score for this model is around 78%. This is a better positivity score compared to Sampling score of 66%
-   The precision scores for this model at 4% 
-   The recall score for this model is at 67%

![Ensemble_BalancedRandomClassifier](https://user-images.githubusercontent.com/80075982/125714965-c1f40b7a-633a-4af3-826f-6d149ebb8b7f.png)

### Easy Ensemble Classifier
-   The balanced accuracy score for this model is around 93%. This is an excellent positive score.
-   The precision scores for this model at 7% 
-   The recall score for this model is at 91%

![Ensemble_EnsembleAdaBooster_Classifier](https://user-images.githubusercontent.com/80075982/125714967-4c7fe2ba-d33e-4bc3-b9bf-30eb11f49816.png)

## Summary
Comparing the 4 sampling algorithms: Oersampling-Naive Random, Oversampling-SMOTE, Undersampling and Combination Over-Under sampling, balanced accuracy at 66%. Precision score is very low positivity and recall score is fairly positive but not excelent. 

The ensemble algorithms: balanced random forest classifier and easy ensemble Classifier much better than the Sampling models. Balanced accuracy at 78% for the Balanced Random Forest Classifier and 91% for the Easy Ensemble Classifier.

Of the models created, the Easy Ensemble AdaBoost Classifier would be the best model to use to predict credit risk due to the high recall scores for both high and low risk loans, as well as an accuracy score of 92.5%. The precision for this model is still very off, indicating that the positives are not necessarily accurate, and so this model could be much improved training and testing more data before putting it into use.

**Predicting U.S. Airlines' Domestic Flight Delay and Cancellation (January 2019 - August 2023)**

  **Project Overview**

This project was developed as part of **Capstone Project 2** for the Bachelor of Science (Honours) in Information Systems (Data Analytics) at Sunway University. The objective is to develop machine learning models capable of predicting U.S. domestic flight arrival delay, departure delays, and flight cancellations using only pre-flight information. The project evaluates multiple regression and classification algorithms and compares their predictive performance using standard evaluation metrics.

_________________________________________________________________________________________________________________

 **Dataset**

 **Source:** Bureau of Transportation Statistics (BTS)

 **Coverage:**

-
    U.S. domestic flights
-
    January 2019 - August 2023

_________________________________________________________________________________________________________________

**Project Objectives**

-
  Predict arrival delay using regression models.

-
  Predict departure delay using regression models.

-
  Predict flight delays (greater than 15 minutes) using classification models.

-
  Predict flight cancellations using binary classification models.

-
  Compare the performance of multiple machine learning algorithms

________________________________________________________________________________________________________________

**Machine Learning Models**

**Regression Models**

-
  Linear Regression

-
  Decision Tree Regressor

-
  Gradient Boosting Regressor

-
  Random Forest Regressor

-
  XGBoost Regressor

**Classification Models**

-
  Logistic Regression

-
  Decision Tree Classifier

-
  Gradient Boosting Classifier

-
  Random Forest Classifier

________________________________________________________________________________________________________________

**Data Preparation**

The pre-processing pipeline includes:

-
  Extracts year, month, and day from the flight date

-
  Converts flight times from HHMMM format to minutes

-
  Removes unnecessary indentification columns

-
  Remove variables that may cause data leakage

-
  Excludes cancelled and diverted flights rom delay modelling

-
  Applies frequency encoding to categorical variables

-
  Applies median imputation to missing numerical values

-
  Applies feature scaling for classification models

-
  Splits the data into 80% training and 20% testing sets

-
  Uses stratified sampling for cancellation prediction

Class imbalance is handled using class weights, sample weights, and scale_pos_weight.

______________________________________________________________________________________________________________________________

**Evaluation Metrcis**

**Regression**

-
  Root Mean Squared Error

-
  Mean Absolute Error

-
  R^2 Score

**Classification**

-
  Accuracy

-
  Precision

-
  Recall

-
  F1-score

-
  ROC-AUC

The best regression model is selected using the test R^2. The best classification model is selected using test F1-score.

**Outputs**

The script generates:

-
  Model performance CSV files

-
  Dataset descriptive statistics

-
  Leakage exclusion table

-
  Cancellation class balance chart

-
  Delay distribution histograms

-
  Correlation heatmaps

-
  Model comparison charts

-
  Train and test performance charts

-
  Actual vs. Predicted plots

-
  Residual plots

-
  Feature importance charts

-
  Confusion matrices

-
  ROC curves

**Required Libraries**

-
  pandas

-
  numpy

-
  matplotlib

-
  seaborn

-
  scipy

-
  scikit-learn

______________________________________________________________________________________________________________________________

  **Author**

  **Muhammad Talal Naveed**

  BSc Information Systems (Honours) (Data Analytics)

  Sunway University

  Capstone Project 2

  **Supervisor:** Dr.Ghulam Murtaza

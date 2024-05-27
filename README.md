# Heart--Disease--Prediction--with--Logistic--Regression

# Instruction

## Problem Statement:-

One of the hospitals has a patient dataset that contains a wide range of heart-related features. This data allows hospital staff to conduct detailed analyses of heart-related conditions and treatments. We must build a logistic regression model to predict whether a patient has heart disease or not. Calculate the feature importance as well. The dataset contains data for around 303 patients.

## Data Description:-

**age:** Age of the patient in years.
**gender:** Gender of the patient.
**cp:** Chest pain type.
**trestbps:** Resting blood pressure (in mm Hg on admission to the hospital).
**chol:** Serum cholesterol in mg/dl.
**fbs:** fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
**restecg:** Resting electrocardiographic results.
**thalach:** Maximum heart rate achieved.
**exang:** Exercise induced angina (1 = yes; 0 = no).
**oldpeak:** ST depression induced by exercise relative to rest.
**slope:** The slope of the peak exercise ST segment.
**ca:** Number of major vessels (0-3) colored by fluoroscopy.
**thal:** 3 = normal; 6 = fixed defect; 7 = reversible defect.
**heart_diagnosis:** Diagnosis of heart disease (angiographic disease status) (0 = No heart disease, >0 = heart disease).

## Tasks/Activities List:-

Your code should contain the following activities/Analysis:

* Read the heart dataset.
* Exploratory Data Analysis (EDA) - Show the Data quality check, treat the missing values, etc if any.
* Transform the categorical data.
* Apply the Logistic Regression model.
* Print the model results.
* Get the feature importance.

# The steps followed in developing this model is as follows:-

1. Importing necessary libraries We have used Numpy, Pandas for dataframe, Matplotlib and Seaborn for visualization and SKLearn for TrainTest Split,Model evaluation, metrix for building the LRM.
2. Importing and Describing After importing the dataset, 7 columns having null values in dataset. This columns are int/float datatype so null values replace with median of eavh column respectively. Finally, we have a Dataset with zero null values.
3. some columns are having outliers, so outliers are handled by statistical method. Lower than lower fense replace with lower fense and higher than higher fense replace with higher fense. now no any outliers in dataset.
4. Correation and exploratory data analsysis done on dataset. some of insights found in dataset.
5. Splitting the data into Train and Test Data with train size 80% for Training and Testing purpose. This is done using SKLearn’s train_test_split function.
6. Modularize machine learning model developed by using logistic regression. 8 no's columns selected for building model which are having some relations with dependent variable.
7. Evaluating the model We determine the Confusion Matrix and the parameters like accuracy score etc. Accuracy score of model is 87%.

# Insights of dataset:-

### Data Analysis Steps

1. **Exploratory Data Analysis (EDA)**:

   - Summary statistics
   - Distribution plots for numerical variables
   - Frequency counts for categorical variables
2. **Data Cleaning**:

   - Check for missing values
   - Handle outliers if necessary
   - Convert categorical variables to appropriate formats
3. **Feature Engineering**:

   - Create new features if necessary
   - Transform or bin certain variables for better analysis or model performance
4. **Model Building**:

   - Split the data into training and testing sets
   - Train various models (e.g., Logistic Regression, Decision Trees, Random Forest, SVM)
   - Evaluate models using appropriate metrics (e.g., accuracy, precision, recall, F1-score)
5. **Model Evaluation**:

   - Use cross-validation to ensure model robustness
   - Perform hyperparameter tuning for the best model performance
6. **Interpretation and Reporting**:

   - Interpret the model results
   - Report the findings with visualizations and statistical insights

Let’s begin by performing some basic exploratory data analysis (EDA) to get an initial understanding of the dataset.

Here's a summary and analysis of the provided data on heart disease diagnosis:

**Summary:**
The dataset contains 304 entries with various attributes related to heart health, including:

- **Age** (years)
- **Gender** (1 = male, 0 = female)
- **Chest Pain Type (cp)**:
  - 1: Typical angina
  - 2: Atypical angina
  - 3: Non-anginal pain
  - 4: Asymptomatic
- **Resting Blood Pressure (trestbps)**: in mm Hg
- **Serum Cholesterol (chol)**: in mg/dl
- **Fasting Blood Sugar (fbs)**: > 120 mg/dl (1 = true; 0 = false)
- **Resting Electrocardiographic Results (restecg)**:
  - 0: Normal
  - 1: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
  - 2: Showing probable or definite left ventricular hypertrophy by Estes' criteria
- **Maximum Heart Rate Achieved (thalach)**
- **Exercise Induced Angina (exang)**: 1 = yes; 0 = no
- **Oldpeak**: ST depression induced by exercise relative to rest
- **Slope**: The slope of the peak exercise ST segment
  - 1: Upsloping
  - 2: Flat
  - 3: Downsloping
- **Number of Major Vessels (ca)**: colored by fluoroscopy (0-3)
- **Thal**:
  - 3: Normal
  - 6: Fixed defect
  - 7: Reversible defect
- **Heart Diagnosis (heart_diagnosis)**: 0 = no heart disease, 1 = heart disease (1-4 scale)

**Analysis:**

1. **Gender Distribution**:

   - Males: 207
   - Females: 97
2. **Age Distribution**:

   - Range: 29 to 77 years
   - Median: 56 years
3. **Chest Pain Type**:

   - 1 (Typical angina): 23
   - 2 (Atypical angina): 50
   - 3 (Non-anginal pain): 86
   - 4 (Asymptomatic): 145
4. **Resting Blood Pressure**:

   - Range: 94 to 200 mm Hg
   - Median: 130 mm Hg
5. **Serum Cholesterol**:

   - Range: 141 to 564 mg/dl
   - Median: 240 mg/dl
6. **Fasting Blood Sugar**:

   - FBS > 120 mg/dl: 45 (15%)
   - FBS ≤ 120 mg/dl: 259 (85%)
7. **Resting Electrocardiographic Results**:

   - Normal: 147
   - ST-T wave abnormality: 91
   - Left ventricular hypertrophy: 66
8. **Maximum Heart Rate Achieved**:

   - Range: 88 to 202 bpm
   - Median: 153 bpm
9. **Exercise Induced Angina**:

   - Yes: 109
   - No: 195
10. **Oldpeak**:

    - Range: 0 to 6.2
    - Median: 1
11. **Slope of Peak Exercise ST Segment**:

    - Upsloping: 99
    - Flat: 138
    - Downsloping: 67
12. **Number of Major Vessels Colored by Fluoroscopy**:

    - 0 vessels: 178
    - 1 vessel: 73
    - 2 vessels: 36
    - 3 vessels: 17
13. **Thalassemia**:

    - Normal: 166
    - Fixed defect: 18
    - Reversible defect: 120
14. **Heart Disease Diagnosis**:

    - No heart disease (0): 160
    - Heart disease (1-4): 144

**Key Insights:**

- **Gender and Heart Disease**: Males have a higher frequency of heart disease diagnosis compared to females.
- **Age**: Older individuals tend to have a higher prevalence of heart disease.
- **Chest Pain**: Asymptomatic individuals (cp = 4) represent the largest group, suggesting silent or unrecognized heart issues.
- **Exercise Induced Angina**: A significant number of patients with heart disease experience exercise-induced angina.
- **Thalassemia**: Patients with reversible defects have a higher incidence of heart disease.

The dataset provides a comprehensive view of the various factors contributing to heart disease, emphasizing the importance of multiple diagnostic criteria in assessing cardiovascular health..

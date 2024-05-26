# Heart-disease----Prediction-with-Logistic-Regression
Need of machine learning model which will predict heart disease on multiple independent variable. To solve this problem logistic regression model is developed which predicts a heart disease with 86% accuracy score.

# The steps followed in developing this model is as follows:-
1. Importing necessary libraries We have used Numpy, Pandas for dataframe, Matplotlib and Seaborn for visualization and SKLearn for TrainTest Split,Model evaluation, metrix for building the LRM.

2. Importing and Describing After importing the dataset, 7 columns having null values in dataset. This columns are int/float datatype so null values replace with median of eavh column respectively. Finally, we have a Dataset with zero null values.

3. some columns are having outliers, so outliers are handled by statistical method. Lower than lower fense replace with lower fense and higher than higher fense replace with higher fense. now no any outliers in dataset.

4. Correation and exploratory data analsysis done on dataset. some of insights found in dataset.

5. Splitting the data into Train and Test Data with train size 80% for Training and Testing purpose. This is done using SKLearn’s train_test_split function.

5. Modularize machine learning model developed by using logistic regression. 8 no's columns selected for building model which are having some relations with dependent variable. 

6. Evaluating the model We determine the Confusion Matrix and the parameters like accuracy score etc. Accuracy score of model is 86%.

# Insights of dataset - 
- Heart disease study done on 4238 no's of peoples - Males are 43% and females are 57%.
- out of 4238 No's poeples - Heart disease cases are 644 No's- 343 No's Male and 301 No's Female - 8% male and 7% female
- heart disease is more in male then female.
- Max heart disease cases in age - Male 50 to 52 years and female 63 to 64 years.
- Education category 1.0 having max heart disease cases.
- Male consumed more cigs per day than female
- even with less cigs consumption, female having nearly same heart disease cases. so cigs conumption affect more on male than female.
- No any relation of current smoker,prevalent stroke,prevalentHyp,BPMeds and diabeties with heart disease.
- max heart disease cases fall under cholestrol range of 220 to 260,MBI range 23.5 to 27.5, sysBP range 125 to 145.
- Max heart disease cases fall under diaBP range 80 to 90, heart rate range 73 to 76, glucose range 78 to 81.
# Heart--Disease--Prediction-With--Logistic--Regression

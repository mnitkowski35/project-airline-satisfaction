# Welcome to My Airline Passenger Satisfaction Analysis Presented by Matthew Nitkowski

## Overview

In this project, I am building a Streamlit application that predicts the satisfaction of airline passengers based on a dataset. The dataset used in this project is the "Airline Passenger Satisfaction" dataset, which can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data). The dataset contains information about airline passengers, including features such as flight distance, seat comfort, inflight entertainment, and more.

We will be utilizing libraries such as Pandas, Seaborn, MatplotLib, Numpy, and SciKit Learn.

## Data Cleaning and EDA

I first noticed that there was an unnamed column with no data, so iI removed it.

There was also a category, Arrival Delay in Minutes, with null values. Since it was numerical, I filled in the null values with the mean of the entire column.

Those were the only data cleaning steps I did. However, I did need to make a few adjustments in my EDA for some categorical variables. I converted them to numbers so I could use them in my predictive models.

> I got the dummies of categorical variables with two values: Gender, Customer Type, and Type of Travel.

> For Class and Satisfaction, I mapped them because I wanted to specify which values were which. For class, Business was 0, Eco was 1, and Eco Plus was 2. For satisfaction, satisfied was 1 and neutral_or_dissatisfied was 0, as satisfaction is our target variable so I wanted satisfied to be 1. Getting the dummies of the target variable column was not ideal.

I then built a heatmap to see which variables had the strongest correlation to heatmap. Once that was done, I built Linear Regression, Random Forest Classifier, K-Nearest-Neighbors, and Logistic Regression prediction models from a handful of features I selected. I used all models but Logistic Regression in my final draft.

# Conclusion

The variables that had the most impact on my model were online boarding, class, and type of travel. The strongest model was the Random Forest Classifier, with an accuracy of 95.9 percent. Our baseline model accuracy was 43.3 percent.
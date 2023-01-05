# Predicting Credit Card Approvals with Logistic Regression

In this project we are going to build a Machine Learning model using the Logistic Regression algorithm, to predict whether a request for a credit card gets rejected or approved. There are various factors determining the result of a credict card request, namely high loan balances, low income levels, or too many inquiries on an individual's credit report. We are going to use all these features to build an automatic credit card approval predictor using machine learning.

![image](https://user-images.githubusercontent.com/113103161/210847647-26a2f4ba-fd06-45f7-bd91-0aec8308d66e.png)

## Project Outline
- First, we will start off by loading and viewing the dataset.
- We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.
- We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
- After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
- Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.


## Project Tasks
1. Credit card applications
2. Inspecting the applications
3. Splitting the dataset into train and test sets
4. Handling the missing values (part i)
5. Handling the missing values (part ii)
6. Handling the missing values (part iii)
7. Preprocessing the data (part i)
8. Preprocessing the data (part ii)
9. Fitting a logistic regression model to the train set
10. Making predictions and evaluating performance
11. Grid searching and making the model perform better
12. Finding the best performing model


### Credit card applications
First we load our dataset into ```cc_apps``` using  ```pandas```.

### Inspecting the applications

The loaded dataset includes the following: Gender, Age, Debt, Married, BankCustomer, EducationLevel, Ethnicity, YearsEmployed, PriorDefault, Employed, CreditScore, DriversLicense, Citizen, ZipCode, Income and finally the ApprovalStatus.


```
'data.frame':   689 obs. of  16 variables:
 $ Male          : num  1 1 0 0 0 0 1 0 0 0 ...
 $ Age           : chr  "58.67" "24.50" "27.83" "20.17" ...
 $ Debt          : num  4.46 0.5 1.54 5.62 4 ...
 $ Married       : chr  "u" "u" "u" "u" ...
 $ BankCustomer  : chr  "g" "g" "g" "g" ...
 $ EducationLevel: chr  "q" "q" "w" "w" ...
 $ Ethnicity     : chr  "h" "h" "v" "v" ...
 $ YearsEmployed : num  3.04 1.5 3.75 1.71 2.5 ...
 $ PriorDefault  : num  1 1 1 1 1 1 1 1 1 0 ...
 $ Employed      : num  1 0 1 0 0 0 0 0 0 0 ...
 $ CreditScore   : num  6 0 5 0 0 0 0 0 0 0 ...
 $ DriversLicense: chr  "f" "f" "t" "f" ...
 $ Citizen       : chr  "g" "g" "g" "s" ...
 $ ZipCode       : chr  "00043" "00280" "00100" "00120" ...
 $ Income        : num  560 824 3 0 0 ...
 $ Approved      : chr  "+" "+" "+" "+" ...
 
 ```
Now, we inspect the structure, numerical summary, and specific rows of the dataset by extracting the summary statistics of the data using the ```describe()``` method of ```cc_apps```. Then, we use the ```info()``` method of ```cc_apps``` to get more information about the DataFrame.

### Splitting the dataset into train and test sets

### Handling the missing values (part i)

### Handling the missing values (part ii)

### Handling the missing values (part iii)

### Preprocessing the data (part i)

### Preprocessing the data (part ii)

### Fitting a logistic regression model to the train set

### Making predictions and evaluating performance

### Grid searching and making the model perform better

### Finding the best performing model

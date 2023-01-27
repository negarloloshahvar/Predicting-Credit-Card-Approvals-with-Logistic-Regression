# Predicting Credit Card Approvals with Logistic Regression

In this project we are going to build a Machine Learning model using the Logistic Regression algorithm, to predict whether a request for a credit card gets rejected or approved. There are various factors determining the result of a credict card request, namely high loan balances, low income levels, or too many inquiries on an individual's credit report. We are going to use all these features to build an automatic credit card approval predictor using machine learning.


![image](https://images.unsplash.com/photo-1609429019995-8c40f49535a5?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2069&q=80)

## Project Outline
- First, we will start off by loading and viewing the dataset.
- We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.
- We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
- After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
- Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.


## Project Tasks
1. [Credit card applications](#1.-Credit-card-applications)
2. [Inspecting the applications](#2.-Inspecting-the-applications)
3. [Splitting the dataset into train and test sets](#3.-Splitting-the-dataset-into-train-and-test-sets)
4. [Handling the missing values](#4.-Handling-the-missing-values)
5. [Preprocessing the data](#5.-Preprocessing-the-data)
6. [Fitting a logistic regression model to the train set](#6.-Fitting-a-logistic-regression-model-to-the-train-set)
7. [Making predictions and evaluating performance](#7.-Making-predictions-and-evaluating-performance)
8. [Grid searching and making the model perform better](#8.-Grid-searching-and-making-the-model-perform-better)
9. [Finding the best performing model](#9.-Finding-the-best-performing-model)


### 1. Credit card applications
First we load our dataset into ```cc_apps``` using  ```pandas```. The loaded dataset includes the following: Gender, Age, Debt, Married, BankCustomer, EducationLevel, Ethnicity, YearsEmployed, PriorDefault, Employed, CreditScore, DriversLicense, Citizen, ZipCode, Income and finally the ApprovalStatus.


```python
import pandas as pd
cc_apps = pd.read_csv('Dataset/cc_approvals.data', header= None)
cc_apps.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>30.83</td>
      <td>0.000</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.25</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>00202</td>
      <td>0</td>
      <td>+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>58.67</td>
      <td>4.460</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>3.04</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
      <td>f</td>
      <td>g</td>
      <td>00043</td>
      <td>560</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>24.50</td>
      <td>0.500</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>1.50</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00280</td>
      <td>824</td>
      <td>+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>27.83</td>
      <td>1.540</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>3.75</td>
      <td>t</td>
      <td>t</td>
      <td>5</td>
      <td>t</td>
      <td>g</td>
      <td>00100</td>
      <td>3</td>
      <td>+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>20.17</td>
      <td>5.625</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.71</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>s</td>
      <td>00120</td>
      <td>0</td>
      <td>+</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Inspecting the applications
Now, we inspect the structure, numerical summary, and specific rows of the dataset by extracting the summary statistics of the data using the ```describe()``` method of ```cc_apps```. Then, we use the ```info()``` method of ```cc_apps``` to get more information about the DataFrame.

<a id='2._Inspecting_the_applications'></a>


```python
print(cc_apps.info())
print('\n', cc_apps.describe())
print('\n', cc_apps.tail(17))
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 690 entries, 0 to 689
    Data columns (total 16 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   0       690 non-null    object 
     1   1       690 non-null    object 
     2   2       690 non-null    float64
     3   3       690 non-null    object 
     4   4       690 non-null    object 
     5   5       690 non-null    object 
     6   6       690 non-null    object 
     7   7       690 non-null    float64
     8   8       690 non-null    object 
     9   9       690 non-null    object 
     10  10      690 non-null    int64  
     11  11      690 non-null    object 
     12  12      690 non-null    object 
     13  13      690 non-null    object 
     14  14      690 non-null    int64  
     15  15      690 non-null    object 
    dtypes: float64(2), int64(2), object(12)
    memory usage: 86.4+ KB
    None
    
                    2           7          10             14
    count  690.000000  690.000000  690.00000     690.000000
    mean     4.758725    2.223406    2.40000    1017.385507
    std      4.978163    3.346513    4.86294    5210.102598
    min      0.000000    0.000000    0.00000       0.000000
    25%      1.000000    0.165000    0.00000       0.000000
    50%      2.750000    1.000000    0.00000       5.000000
    75%      7.207500    2.625000    3.00000     395.500000
    max     28.000000   28.500000   67.00000  100000.000000
    
         0      1       2  3  4   5   6      7  8  9   10 11 12     13   14 15
    673  ?  29.50   2.000  y  p   e   h  2.000  f  f   0  f  g  00256   17  -
    674  a  37.33   2.500  u  g   i   h  0.210  f  f   0  f  g  00260  246  -
    675  a  41.58   1.040  u  g  aa   v  0.665  f  f   0  f  g  00240  237  -
    676  a  30.58  10.665  u  g   q   h  0.085  f  t  12  t  g  00129    3  -
    677  b  19.42   7.250  u  g   m   v  0.040  f  t   1  f  g  00100    1  -
    678  a  17.92  10.210  u  g  ff  ff  0.000  f  f   0  f  g  00000   50  -
    679  a  20.08   1.250  u  g   c   v  0.000  f  f   0  f  g  00000    0  -
    680  b  19.50   0.290  u  g   k   v  0.290  f  f   0  f  g  00280  364  -
    681  b  27.83   1.000  y  p   d   h  3.000  f  f   0  f  g  00176  537  -
    682  b  17.08   3.290  u  g   i   v  0.335  f  f   0  t  g  00140    2  -
    683  b  36.42   0.750  y  p   d   v  0.585  f  f   0  f  g  00240    3  -
    684  b  40.58   3.290  u  g   m   v  3.500  f  f   0  t  s  00400    0  -
    685  b  21.08  10.085  y  p   e   h  1.250  f  f   0  f  g  00260    0  -
    686  a  22.67   0.750  u  g   c   v  2.000  f  t   2  t  g  00200  394  -
    687  a  25.25  13.500  y  p  ff  ff  2.000  f  t   1  t  g  00200    1  -
    688  b  17.92   0.205  u  g  aa   v  0.040  f  f   0  f  g  00280  750  -
    689  b  35.00   3.375  u  g   c   h  8.290  f  f   0  t  g  00000    0  -
    

### 3. Splitting the dataset into train and test sets

Taking a good look at the data, we understand that features such as ```DriverLisence``` or ```ZipCode``` are not effective in credir approval and we can set them aside using the ```drop()``` method. Next, it is time to split our data into train set and test set.


```sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)```


```python
from sklearn.model_selection import train_test_split

cc_apps = cc_apps.drop([11, 13], axis= 1)

cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size= 0.33, random_state= 42)

```


```python
cc_apps
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>12</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>30.83</td>
      <td>0.000</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.25</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>g</td>
      <td>0</td>
      <td>+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>58.67</td>
      <td>4.460</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>3.04</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
      <td>g</td>
      <td>560</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>24.50</td>
      <td>0.500</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>1.50</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>g</td>
      <td>824</td>
      <td>+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>27.83</td>
      <td>1.540</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>3.75</td>
      <td>t</td>
      <td>t</td>
      <td>5</td>
      <td>g</td>
      <td>3</td>
      <td>+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>20.17</td>
      <td>5.625</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.71</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>s</td>
      <td>0</td>
      <td>+</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>685</th>
      <td>b</td>
      <td>21.08</td>
      <td>10.085</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>1.25</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>g</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>686</th>
      <td>a</td>
      <td>22.67</td>
      <td>0.750</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>2.00</td>
      <td>f</td>
      <td>t</td>
      <td>2</td>
      <td>g</td>
      <td>394</td>
      <td>-</td>
    </tr>
    <tr>
      <th>687</th>
      <td>a</td>
      <td>25.25</td>
      <td>13.500</td>
      <td>y</td>
      <td>p</td>
      <td>ff</td>
      <td>ff</td>
      <td>2.00</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>g</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>688</th>
      <td>b</td>
      <td>17.92</td>
      <td>0.205</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.04</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>g</td>
      <td>750</td>
      <td>-</td>
    </tr>
    <tr>
      <th>689</th>
      <td>b</td>
      <td>35.00</td>
      <td>3.375</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>h</td>
      <td>8.29</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>g</td>
      <td>0</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
<p>690 rows × 14 columns</p>
</div>



### 4. Handling the missing values
No dataset is perfect and this dataset is not an exception! First of all, we can observe that there are many missing values that are shown as '?'. We can replace these question marks with np.NaN from ```numpy``` that makes more sense. We do this by using ```replace()``` function.

``` DataFrame.replace(to_replace=None, value=_NoDefault.no_default, *, inplace=False, limit=None, regex=False, method=_NoDefault.no_default)```


```python
import numpy as np
cc_apps_train = cc_apps_train.replace('?', np.NaN)
cc_apps_test = cc_apps_test.replace('?', np.NaN)
```

Next, we impute the missing values with a strategy called **mean imputation**. However, this strategy is not a very good one as it ignores all the features correlations.

In mean imputation, we replace all the null values with the mean of its column. to do this, we use pandas ```fillna()``` function to replace the missing values with their corresponding mean calculated by ```np.mean()```. We must pay attantion to the fact that the ```fillna()``` method implicitly handles the imputations for the columns containing **numeric** data-types. 


```python
cc_apps_train.fillna(cc_apps_train.mean(), inplace=True)
cc_apps_test.fillna(cc_apps_test.mean(), inplace=True)


print(cc_apps_train.isnull().sum())
```

    0     8
    1     5
    2     0
    3     6
    4     6
    5     7
    6     7
    7     0
    8     0
    9     0
    10    0
    12    0
    14    0
    15    0
    dtype: int64
    

    C:\Users\negar\AppData\Local\Temp\ipykernel_24924\2849595082.py:1: FutureWarning: The default value of numeric_only in DataFrame.mean is deprecated. In a future version, it will default to False. In addition, specifying 'numeric_only=None' is deprecated. Select only valid columns or specify the value of numeric_only to silence this warning.
      cc_apps_train.fillna(cc_apps_train.mean(), inplace=True)
    C:\Users\negar\AppData\Local\Temp\ipykernel_24924\2849595082.py:2: FutureWarning: The default value of numeric_only in DataFrame.mean is deprecated. In a future version, it will default to False. In addition, specifying 'numeric_only=None' is deprecated. Select only valid columns or specify the value of numeric_only to silence this warning.
      cc_apps_test.fillna(cc_apps_test.mean(), inplace=True)
    

We are going to impute non-numeric missing values with the most frequent values as present in the respective columns. This is good practice when it comes to imputing missing values for categorical data in general.


```python
for col in cc_apps_train.columns:
    if cc_apps_train[col].dtypes == 'object':
        cc_apps_train = cc_apps_train.fillna(cc_apps_train[col].value_counts().index[0])
        cc_apps_test = cc_apps_test.fillna(cc_apps_train[col].value_counts().index[0])
        
print(cc_apps_train.isna().sum())
```

    0     0
    1     0
    2     0
    3     0
    4     0
    5     0
    6     0
    7     0
    8     0
    9     0
    10    0
    12    0
    14    0
    15    0
    dtype: int64
    

### 5. Preprocessing the data
Now that the missing data is handled, we take care of the following:
- Converting the non-numeric data into numeric.
- Scaling the feature values to a uniform range.


```python
#categorical data#
cc_apps_train = pd.get_dummies(cc_apps_train)
cc_apps_test = pd.get_dummies(cc_apps_test)

cc_apps_test = cc_apps_test.reindex(columns = cc_apps_train.columns, fill_value = 0)
```


```python
# feature scaling
from sklearn.preprocessing import MinMaxScaler

X_train, y_train = cc_apps_train.iloc[:, :-1].values, cc_apps_train.iloc[:, [-1]].values.ravel()
X_test, y_test = cc_apps_test.iloc[:, :-1].values, cc_apps_test.iloc[:, [-1]].values.ravel()

scaler = MinMaxScaler(feature_range= (0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)
```

### 6. Fitting a logistic regression model to the train set


```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(rescaledX_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>



### 7. Making predictions and evaluating performance


```python
from sklearn.metrics import confusion_matrix

y_pred = logreg.predict(rescaledX_test)

print(f'Model Accuracy= {logreg.score(rescaledX_test, y_test)}')
print(confusion_matrix(y_test, y_pred))
```

    Model Accuracy= 1.0
    [[103   0]
     [  0 125]]
    

### 8. Grid searching and making the model perform better

The model works perfectly, however, we can implement a Grid Search to improve our hyperparameters. We will grid search over the following two:

- tol
- max_iter


```python
from sklearn.model_selection import GridSearchCV

tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

param_grid = dict(tol= tol, max_iter= max_iter)
```

### 9. Finding the best performing model


```python
grid_model = GridSearchCV(estimator= logreg, param_grid= param_grid, cv= 5)
grid_model_results = grid_model.fit(rescaledX_train, y_train)
best_score= grid_model_results.best_score_
best_param= grid_model_results.best_params_
print(f'Best {best_score} using {best_param}')

best_model = grid_model_results.best_estimator_
print(f'Accuracy of the classifier is {best_model.score(rescaledX_test, y_test)}')
```

    Best 1.0 using {'max_iter': 100, 'tol': 0.01}
    Accuracy of the classifier is 1.0
    

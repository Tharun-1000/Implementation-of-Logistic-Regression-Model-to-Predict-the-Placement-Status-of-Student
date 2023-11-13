# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.


3.Import LabelEncoder and encode the dataset.


4.Import LogisticRegression from sklearn and apply the model on the dataset. 


5.Predict the values of array. 


6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.


7.Apply new unknown values 

## Program:

Developed by: THARUN K

RegisterNumber:  212222040172

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#remove the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



## Output:

1.Placement Data

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/977247be-6200-4a6a-a27d-629dc9886e5c)

2.Salary Data

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/db6bac71-f9a4-4b5d-b296-24bb955d14b0)

3.Checking the null() function

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/b81001fb-29ed-4499-88e3-e132552bb277)

4.Data Duplicate

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/2fe0a4e9-87ae-40e2-8e51-d384a9d80692)

5.Print Data

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/59c00845-b07e-4bb1-bb4c-d078926372dc)

6.Data-status

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/8e7f1cec-6d23-4ffa-9cd0-4b77d18dc471)

7.y_prediction array

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/a0b338cc-35bf-467e-ab96-48e7ddf51975)

8.Accuracy Value

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/23ae429f-29e6-494d-ae89-9b6698392b8c)

9.Confusion Array

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/fcd3ce67-4534-4a58-a38d-ac8fe61f4b4c)

10.Classification Report

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/038d99c5-150b-474a-8bb7-3f5e2a2836a1)

11.Prediction of LR

![image](https://github.com/Tharun-1000/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135952958/886a6912-0e59-4204-b71c-11eb52c90aac)
















## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student. Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

Import the standard libraries.
Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
LabelEncoder and encode the dataset.
Import LogisticRegression from sklearn and apply the model on the dataset.
Predict the values of array.
Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
Apply new unknown values

# Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Agalya R
RegisterNumber:  212222040003
*/
import pandas as pd
data=pd.read_csv('/Placement_Data(1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)#Accuracy Score = (TP+TN)/(TP+FN+TN+FP)
#accuracy_score(y_true,y_pred,normalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion #11+24=35 -correct predictions,5+3=8 incorrect predictions

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

# Output:
# Placement data
![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/7bdc3810-5c9b-4a4f-ad22-29d8dffc83a5)


# Salary data
![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/b312edac-7a9d-47a4-98e6-c8c7cacc598e)


# Checking the null() function
![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/ae7d4f72-fe46-4eda-a7f6-96a37841746f)


# Data Duplicate

![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/cd7e7953-53cc-4952-bc29-79680c1b4ca7)


# Print data

![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/075652d2-6bc5-4807-bb3a-4f94047483e5)


# Data-status

![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/c58a55c8-df7d-4e32-ba69-c99222e4c382)


# y_prediction array

![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/f1fb11c0-0da8-4e3d-9489-724fff8ca44e)


# Accuracy value

![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/db9af4c2-95ff-4513-aef6-6dd11766505e)


# Confusion array

![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/a2222a19-b841-426e-a176-853eb0d163dd)


# Classification report

![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/2bddc10b-08c6-451b-82f3-f7e0b82f3102)


# Prediction of LR

![image](https://github.com/gracia55/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129026838/3f17ccb6-2da7-4ab7-849e-edfcca3d3d20)


# Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

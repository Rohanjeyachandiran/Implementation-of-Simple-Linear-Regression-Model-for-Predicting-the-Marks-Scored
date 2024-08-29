# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe
4. Plot the required graph both for test data and training data.
5.Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sushiendar M
RegisterNumber: 212223040217
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
 
Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:
![EXPNO2OUTPUT1](https://github.com/user-attachments/assets/f4cd7c25-1c92-49ea-8c5b-846720403a7b)
![2024-08-29 (5)](https://github.com/user-attachments/assets/f9c894e3-45ff-4cae-adff-817e4b9b96a7)
![EXP2OUTPUT3](https://github.com/user-attachments/assets/d4e6ac1f-806e-4432-907f-ae08c6ebea85)
![EXP2OUTPUT4](https://github.com/user-attachments/assets/d8ce8414-84c1-4687-a66d-fd8a8bc60dd0)
![2024-08-29 (13)(1)(1)](https://github.com/user-attachments/assets/e3689fac-9977-47fa-a568-bac887845cf9)
![2024-08-29 (16)(1)(1)](https://github.com/user-attachments/assets/8bf79116-9aa1-4c82-afcf-1ce21c69db21)
![2024-08-29 (20)(1)(1)](https://github.com/user-attachments/assets/37584402-3166-4cf8-bd43-6c1521b4bd8d)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

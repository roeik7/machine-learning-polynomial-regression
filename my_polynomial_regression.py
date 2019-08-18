# -*- coding: utf-8 -*-
"""
@author: roik
"""

#data processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data set
dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values


#splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)"""
 
#feature scaling 
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""


#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regression=LinearRegression()
linear_regression.fit(x,y)

#fitting polunomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression=PolynomialFeatures(degree=5)
x_poly=polynomial_regression.fit_transform(x)
linear_regression2=LinearRegression()
linear_regression2.fit(x_poly, y)


#comparing the linear regression with the polynomial

#visualising the linear regression results
plt.scatter(x, y, color='red')
plt.plot(x, linear_regression.predict(x), color='blue')
plt.title('Truth or Bluff (linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
print(plt.show)


#visualising the polynomial regression results
x_grid=np.arange(min(x), max(x), 0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, linear_regression2.predict(polynomial_regression.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
print(plt.show)

#predict new result with linear regression
print(linear_regression.predict([[6.5]]))

#predict new result with polynomial regression
linear_regression2.predict(polynomial_regression.fit_transform([[6.5]]))
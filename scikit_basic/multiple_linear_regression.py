import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

def adj_r2(x,y): # ? A function that returns the adjusted r_squared based on two data variables;
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


""" 
That's a file with the objective of demonstrating multiple variables linear regression with scikit learn, which is 
more professional, but it's far less intuitive for learning purposes.
"""

data = pd.read_csv('1.02.+Multiple+linear+regression.csv')

x = data[['SAT','Rand 1,2,3']]
y = data['GPA']

reg = LinearRegression()
reg.fit(x,y)

reg.coef_ # ? [ 0.00165354(SAT), -0.00826982(Rand 1, 2 ,3)]
reg.intercept_ # ? 0.29603261264909486

r_squared = reg.score(x,y)  # ? R-Squared value;

adj_r_squared = adj_r2(x,y) # ? 0.39203134825134023

f_regression(x,y) # ? F_regression is a method that creates individual feature regression in order to check for their
# ? usefulness.

p_values = f_regression(x,y)[1]
p_values = p_values.round(3) # ? Output:array([0.   , 0.676]), that way random variable is noticeably unuseful.

#! Creating a summary table with the important files;

reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['P-values'] = p_values

scaler = StandardScaler()
scaler.fit(x) # ? Preparing the scaler mechanism, basically obtaining the mean and standard deviation for future use.
x_scaled = scaler.transform(x) # ? Applying the standardization method to the data base;

reg_2 = LinearRegression()
reg_2.fit(x_scaled,y)

reg_2_summary = pd.DataFrame([['Intercept'],['SAT'],['Rand 1,2,3']], columns=['Features']) # ?  Intercept is also called bias;
reg_2_summary['Weights'] = reg_2.intercept_, reg_2.coef_[0], reg_2.coef_[1]

new_data = pd.DataFrame(data=[[1700,2],[1800,1]],columns=['SAT','Rand 1,2,3']) # ? Creating a new data frame to predict a certain SAT;

reg_2.predict(new_data) # ?  This will generate a wrong info because the database is standardized and the expected output isn't;

new_data_scaled = scaler.transform(new_data) # ? Transforming the new database into a scaled one;
reg_2.predict(new_data_scaled)
reg_simples = LinearRegression()
x_simple_matrix = x_scaled[:,0].reshape(-1,1)
reg_simples.fit(x_simple_matrix,y)
reg_simples.predict(new_data_scaled[:,0].reshape(-1,1))

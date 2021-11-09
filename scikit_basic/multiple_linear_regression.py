import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

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
p_values.round(3) # ? Output:array([0.   , 0.676]), that way random variable is noticeably unuseful.

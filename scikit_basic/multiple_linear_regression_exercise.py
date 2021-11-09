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

data = pd.read_csv('real_estate_price_size_year_2.csv')

x = data[['size','year']]
y = data['price']

reg = LinearRegression()
reg.fit(x,y)

interc  = reg.intercept_ # ? Interception value for the multiple regression
coefic = reg.coef_ # ? Coefficient of each variable
r_sqrd = reg.score(x,y) # ? R-Squared

adj_r_sqrd = adj_r2(x,y)

# ? Now I'll be performing multiple simple linear regressions in order to check variables usefulness

p_values = f_regression(x,y)[1]
p_values = p_values.round(3) #? array([0.   , 0.357]) the second variable "year" is not important at all.

reg_summary = pd.DataFrame(data = x.columns.values,columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['P-values'] = p_values

"""
  Features  Coefficients  P-values
0     size    227.700854     0.000
1     year   2916.785327     0.357

"""
# ! Study conclusion: year isn't relevant and it should be removed from the model;

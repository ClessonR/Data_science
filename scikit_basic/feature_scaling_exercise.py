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


data = pd.read_csv('real_estate_price_size_year_2.csv')

x = data[['size','year']]
y = data['price']



scaler = StandardScaler() #? Creating a scaler instance
scaler.fit(x) #? Calculating the scaling variables
x_scaled = scaler.transform(x) #? Applying the standardization to the main database

reg = LinearRegression()
reg.fit(x_scaled,y)

r_squared = reg.score(x_scaled,y)
adj_r_squared = adj_r2(x_scaled,y)

new_data = pd.DataFrame(data=[[750,2009]], columns=['size','year']) #? This can also be done by just this: new_data = [[750,2009]]
new_data = scaler.transform(new_data) # ? If I'm working with standardized data it's necessary that every new analysis is also standardized;
predicted = reg.predict(new_data) #? prediciting the value;


coef = reg.coef_
bias = reg.intercept_

p_values = f_regression(x,y)[1] # ! Individual p value calculation

reg_sumarry = pd.DataFrame(data=x.columns.values, columns=['Features'])

reg_sumarry['Coefficients'] = coef
reg_sumarry['P-values'] = p_values


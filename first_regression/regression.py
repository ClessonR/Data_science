#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
data = pd.read_csv('1.01.+Simple+linear+regression.csv')
data.describe()
y = data['GPA']
x1 = data['SAT']

plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1,yhat, lw =4, c='red', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()

x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
result.summary()


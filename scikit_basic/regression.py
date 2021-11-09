import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('1.01.+Simple+linear+regression.csv')
data.head()

x = data['SAT'] # ! Input or feature;
y = data['GPA'] # ! Output or Target;




x_matrix = x.values.reshape(-1,1)
x_matrix.shape

reg = LinearRegression()
print(reg.fit(x_matrix,y))
print(reg.score(x_matrix,y))

print(reg.coef_)
print(reg.intercept_)



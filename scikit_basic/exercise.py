import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('real_estate_price_size.csv')

x = data['size'] # ? Input or feature;
y = data['price'] # ? Output or Target;

# ? Plot schematic;
plt.scatter(x,y) # ? Plotting the scatter plot;
plt.xlabel('SIZE', fontsize = 20)
plt.ylabel('PRICE', fontsize = 20)


x_matrix = x.values.reshape(-1,1)
x_matrix.shape
y_matrix = y.values.reshape(-1,1)


reg = LinearRegression()
print(reg.fit(x_matrix,y_matrix))
print(reg.score(x_matrix,y_matrix)) # ! 0.7447391865847586 R-Squared Value

print(reg.intercept_) # ! The incerpeted value, "b in ax + b';
print(reg.coef_) # ! Coefficient of the independent variable;


y_reg = reg.coef_*x_matrix + reg.intercept_ # ! Creating the Regression LIne based
# ! on the data retrieved;


fig = plt.plot(x,y_reg, lw=4, c = 'red', label ='regression')
plt.show()


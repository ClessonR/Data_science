import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as  plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('1.03.+Dummies.csv')
data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes':1, 'No':0})

# ? Here the regression begins:
y = data['GPA'] # ! Selecting GPA as the dependent variable
x1 = data[['SAT','Attendance']] # ! Selecting SAT and Attendance as independent variables

x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
result.summary()

# ? Plotting the regression
plt.scatter(data['SAT'], y)
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'], yhat_no, lw=2, c="#006837")
fig = plt.plot(data['SAT'], yhat_yes, lw=2, c="#a50026")
plt.xlabel('SAT', fontsize =20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

# ! The SAT of the student which attended to more than 75% of the classes were 0.2226 higher than
#! the gpa of students which not attended.

# ? Creating a data frame base on two students SAT scores;
new_data = pd.DataFrame({'const':1, 'SAT':[1700,1670], 'Attendance':[0,1]})
new_data = new_data[['const', 'SAT', 'Attendance']]

#! Creating predictions about the students;
predictions = result.predict(new_data)

#! Creating another data frame;
predictionsdf = pd.DataFrame({'Predictions':predictions})
joined = new_data.join(predictionsdf)
joined.rename(index={0:'Bob', 1:'Alice'})

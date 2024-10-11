import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

salaries = pd.read_csv('ml-salaries.csv')
#salaries.head(2) #print 2 first lignes

"""METHODE TO EXPLORE YOUR DATA"""
salaries.shape() # to see how many rows, columns
salaries.dtypes() # to see available col and their data type
round(salaries.describe()) # to see a readable summary about the dataset, like averages, min and max
salaries['column Name']
salaries[['column Name 1', 'column Name 2']]

#vizualisation of data
#sns.countplot(data=salaries, x='Department') #
#sns.scatterplot(data=salaries, x='Age', y='Salary') # relation of one col to another
#sns.scatterplot(data=salaries, x='Age', y='Salary', hue='Gender') #to add a category to the vizualisation use 'hue'

#1.Create Features and target (Inputs and output) x and y
features = salaries.drop(['salary', 'department'], axis='columns') #input
target = salaries['Salary'] #output

#y = ax+b linear regression

#2.Import the Linear Regression model
model = LinearRegression() #init model

#3. We train the model.
"""
	Linear Regression model looks for a line that best 
	fits all the points in the dataset.
"""
model.fit(features, target) #input and output

#4. We score the model
"""
	Models can have different default scoring metrics. 
	Linear Regression by default uses something called 
	R-squared - a metric that shows how much of change 
	in the target (Gross salary) can be explained by 
	the changes in features (Age, Tenure, Gender etc.)
"""
model.score(features, target) #input and output
#Careful not to confuse this with accuracy. 

# 5.Let's predict the salary of a new hire
features_col = ['Gender', 'Age', 'Department_code', 'Years_exp', 'Tenure (months)']

hire = [[0, 19, 7, 1, 10]]
model.predict(hire)

#6. Explaining the model
	
model.coef_

pd.concat([pd.DataFrame(features.columns), pd.DataFrame(np.transpose(model.coef_))], axis = 1)

model.intercept_

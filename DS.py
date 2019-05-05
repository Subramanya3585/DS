import pandas as pd
import numpy as np
import csv
import re

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# Convert text file into csv
#with open('../input/auto-mpg.txt') as input_file:
#    lines = input_file.readlines()
#    newLines = []
#    for line in lines:
#        newLine = line.strip().split()
#        newLines.append(newLine)

df = pd.read_csv("../input/auto-mpg.csv", header=None)
df.head()

# Renaming the columns
df.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration',
              'model year', 'origin', 'car name']
df.head()

# Lets split variable 'car name' as 'brand name' and 'model name'
df['brand name'], df['model name'] = df['car name'].str.split(' ',1).str

# Lets group and view the data
df.groupby(['brand name']).sum().head()



####Data Cleaning
# Correct brand name 
df['brand name'] = df['brand name'].str.replace('chevroelt|chevrolet|chevy','chevrolet')
df['brand name'] = df['brand name'].str.replace('maxda|mazda','mazda')
df['brand name'] = df['brand name'].str.replace('mercedes|mercedes-benz|mercedes benz','mercedes')
df['brand name'] = df['brand name'].str.replace('toyota|toyouta','toyota')
df['brand name'] = df['brand name'].str.replace('vokswagen|volkswagen|vw','volkswagen')

df.groupby(['brand name']).sum().head()

df.dtypes#we want to make sure all variable are in actual type.

# Convert horsepower from Object to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df.head()

# check horsepower
df.dtypes


###Descriptive Analysis

df.describe().round(2)

# Fill the 6 missing values of horsepower by mean value
meanhp = df['horsepower'].mean()
df['horsepower'] = df['horsepower'].fillna(meanhp)
df.describe().round(2)

# plot distribution plot to view the distribution of target variable
sns.distplot(df['mpg'])

# Skewness and kurtosis
print("Skewness: %f" %df['mpg'].skew())
print("Kurtosis: %f" %df['mpg'].kurt())

#Since skewness value is +ve, the data are positively skewed or skewed right.
#A negative kurtosis means that your distribution is flatter than a normal curve with the same mean and standard deviation.

##Univariate analysis

# Counts of each brands
plt.figure(figsize=(12,6))
sns.countplot(x = "brand name", data=df)
t = plt.xticks(rotation=45)

# Car Counts Manufactured by countries
fig, ax = plt.subplots(figsize = (12, 6))
sns.countplot(x = df.origin.values, data=df)
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = 'USA'
labels[1] = 'Europe'
labels[2] = 'Japan'
ax.set_xticklabels(labels)
ax.set_title("Cars manufactured by Countries")
plt.show()

# Exploring the range and distribution of numerical Variables 

fig, ax = plt.subplots(6, 2, figsize = (15, 13))
sns.boxplot(x= df["mpg"], ax = ax[0,0])
sns.distplot(df['mpg'], ax = ax[0,1])

sns.boxplot(x= df["cylinders"], ax = ax[1,0])
sns.distplot(df['cylinders'], ax = ax[1,1])

sns.boxplot(x= df["displacement"], ax = ax[2,0])
sns.distplot(df['displacement'], ax = ax[2,1])

sns.boxplot(x= df["horsepower"], ax = ax[3,0])
sns.distplot(df['horsepower'], ax = ax[3,1])

sns.boxplot(x= df["weight"], ax = ax[4,0])
sns.distplot(df['weight'], ax = ax[4,1])

sns.boxplot(x= df["acceleration"], ax = ax[5,0])
sns.distplot(df['acceleration'], ax = ax[5,1])

plt.tight_layout()


###Bivariate Analysis
# Plot Numerical Variables 
plt.figure(1)
f,axarr = plt.subplots(4,2, figsize=(12,11))
mpgval = df.mpg.values

axarr[0,0].scatter(df.cylinders.values, mpgval)
axarr[0,0].set_title('Cylinders')

axarr[0,1].scatter(df.displacement.values, mpgval)
axarr[0,1].set_title('Displacement')

axarr[1,0].scatter(df.horsepower.values, mpgval)
axarr[1,0].set_title('Horsepower')

axarr[1,1].scatter(df.weight.values, mpgval)
axarr[1,1].set_title('Weight')

axarr[2,0].scatter(df.acceleration.values, mpgval)
axarr[2,0].set_title('Acceleration')

axarr[2,1].scatter(df["model year"].values, mpgval)
axarr[2,1].set_title('Model Year')

axarr[3,0].scatter(df.origin.values, mpgval)
axarr[3,0].set_title('Country Mpg')
# Rename x axis label as USA, Europe and Japan
axarr[3,0].set_xticks([1,2,3])
axarr[3,0].set_xticklabels(["USA","Europe","Japan"])

# Remove the blank plot from the subplots
axarr[3,1].axis("off")

f.text(-0.01, 0.5, 'Mpg', va='center', rotation='vertical', fontsize = 12)
plt.tight_layout()
plt.show()


###Multi-vairate Analysis
# Correlation between Numerical Features
corr = df.select_dtypes(include=['float64','int64']).iloc[:,0:].corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr,vmax=1, square=True, annot=True)

# Car Manufactured by Countries (USA,Europe,Japan) and multivariate analysis
valtoreplace = {1:'USA', 2:'Europe', 3:'Japan'}
df['norigin'] = df['origin'].map(valtoreplace)
sns.pairplot(df, hue="norigin")

##some insight 
#
#4 cylinder has better mileage performance than other and most manufactured one.
#8 cylinder engines have low mileage count... ofcourse they focus more on pickup( fast cars).
#5 cylinder, performance wise, competes none neither 4 cylinder nor 6 cylinder.
#Displacement, weight, horsepower are inversely related to mileage.
#More horsepower means low mileage.
#Year on Year Manufacturers have focussed on increasing the mileage of the engines.
#Cars manufactured in Japan majorly focuses more on mileage.

Hypothesis testing

df['cylinders'].value_counts()

So, we can see that the categorical variable cylinders have five different levels with 3, 4, 5, 6 and 8 number of cylinders. Now let’s set the hypothesis for this research question.

Null Hypothesis: There is nothing going on between the variables, there is no relationship between the two variables cylinders and mpg. In other words, it does not matter how many cylinders the car has to accurately to predict the mileage of the car, the mean mpg for all the different levels of cylinders variable are same. In mathematical terms

Alternate Hypothesis: There is something going on between the predictor and target variable, or there is a relationship between the two. In other words, the number of cylinders in car affects the mileage of the car, the mean mpg for different groups of cylinder variable or at least one group mean is different from the other group means. But we don’t know which group mean is different, it might be a group with 3 number of cylinders or 4 number of cylinders or even with 8 number of cylinders. In mathematical terms:


Are the differences among the group means are due to true differences between the group means of the population or it is merely due to sampling variability or by chance?
    
#ANOVA F Test
import statsmodels.formula.api as smf

model = smf.ols(formula='mpg ~ cylinders', data=data)
results = model.fit()
print (results.summary())

#Don’t get overwhelmed by this above result. You only need to check the value for the F-statistic and Prob (F-statistic) values. F-statistic is very high at 172.6 with the very very low p-value. So, we can reject our null hypothesis and conclude that there is a relationship between the categorical predictor variable cylinder (number of cylinders in the car) and quantitative target variable mpg (mileage of the car).
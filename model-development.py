import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path of data
path = 'source/automobileEDA.csv'
df = pd.read_csv(path)

# Linear Regression
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X, Y)
# prediction
Yhat = lm.predict(X)

# intercept value and scope value
lm.intercept_
lm.coef_

# Multiple Linear Regression
#  develop a model using these variables as the predictor variables
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

lm.fit(Z, df['price'])

# Model Evaluation Using Visualization
import seaborn as sns

# visualize highway-mpg as potential predictor variable of price
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0, )
plt.show()

# Residual Plot 残差图
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()  # this result shows that the residuals are not randomly spread around the x-axis,
# maybe a non-linear model is more appropriate for this data


# Multiple Linear Regression
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values", ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# Polynomial Regression and Pipelines
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


x = df['highway-mpg']
y = df['price']
# Here we use a polynomial of the 3rd order (cubic)
f = np.polyfit(x, y, 3)
p = np.poly1d(f)

PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)

# perform a polynomial transform on multiple features.
from sklearn.preprocessing import PolynomialFeatures

# create a PolynomialFeatures object of degree 2
pr = PolynomialFeatures(degree=2)
Z_pr = pr.fit_transform(Z)

# Data Pipelines simplify the steps of processing the data.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# create the pipeline by creating a list of tuples including the name of the model or estimator and its corresponding constructor
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),
         ('model', LinearRegression())]
pipe = Pipeline(Input)

# First, we convert the data type Z to type float to avoid conversion warnings that may appear as a result of StandardScaler taking float inputs.
#
# Then, we can normalize the data, perform a transform and fit the model simultaneously.
Z = Z.astype(float)
pipe.fit(Z, y)
ypipe = pipe.predict(Z)

# Measures for In-Sample Evaluation
# Simple Linear Regression
# highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))

Yhat = lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])

from sklearn.metrics import mean_squared_error

#  compare the predicted results with the actual results:
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

# Multiple Linear Regression
# calculate the R^2
# fit the model
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))

Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

# Polynomial Fit
#  import the function r2_score from the module metrics as we are using a different function
from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
# mse
mean_squared_error(df['price'], p(x))

# Prediction and Decision Making
import matplotlib.pyplot as plt
import numpy as np

# create a new input
new_input = np.arange(1, 100, 1).reshape(-1, 1)
# fit the model
lm.fit(X, Y)

# Produce a prediction
yhat = lm.predict(new_input)
yhat[0:5]

# plot the data
plt.plot(new_input, yhat)
plt.show()

# Simple Linear Regression Model (SLR) vs Multiple Linear Regression Model (MLR)
# Usually, the more variables you have, the better your model is at predicting, but this is not always true.
# Sometimes you may not have enough data, you may run into numerical problems, or many of the variables may not be useful and even act as noise. As a result, you should always check the MSE and R^2.
#
# In order to compare the results of the MLR vs SLR models, we look at a combination of both the R-squared and MSE to make the best conclusion about the fit of the model.
#
# MSE: The MSE of SLR is 3.16x10^7 while MLR has an MSE of 1.2 x10^7. The MSE of MLR is much smaller.
# R-squared: In this case, we can also see that there is a big difference between the R-squared of the SLR and the R-squared of the MLR.
# The R-squared for the SLR (~0.497) is very small compared to the R-squared for the MLR (~0.809).
# This R-squared in combination with the MSE show that MLR seems like the better model fit in this case compared to SLR.
#
# Simple Linear Model (SLR) vs. Polynomial Fit
# MSE: We can see that Polynomial Fit brought down the MSE, since this MSE is smaller than the one from the SLR.
# R-squared: The R-squared for the Polynomial Fit is larger than the R-squared for the SLR, so the Polynomial Fit also brought up the R-squared quite a bit.
# Since the Polynomial Fit resulted in a lower MSE and a higher R-squared, we can conclude that this was a better fit model than the simple linear regression for predicting "price" with "highway-mpg" as a predictor variable.
#
# Multiple Linear Regression (MLR) vs. Polynomial Fit
# MSE: The MSE for the MLR is smaller than the MSE for the Polynomial Fit.
# R-squared: The R-squared for the MLR is also much larger than for the Polynomial Fit.
# Conclusion
# Comparing these three models, we conclude that the MLR model is the best model to be able to predict price from our dataset.
# This result makes sense since we have 27 variables in total and we know that more than one of those variables are potential predictors of the final car price.

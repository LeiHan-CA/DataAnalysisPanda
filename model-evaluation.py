import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import clean data
path = 'source/module_5_auto.csv'
df = pd.read_csv(path)

# only use numeric data
df = df._get_numeric_data()
# print(df.head())

from ipywidgets import interact, interactive, fixed, interact_manual


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    # training data
    # testing data
    # lr:  linear regression object
    # poly_transform:  polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])

    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()


# An important step in testing your model is to split your data into training and testing data
y_data = df['price']
# y_data = df['price']
x_data = df.drop('price', axis=1)

# randomly split our data into training and testing data using the function train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])

from sklearn.linear_model import LinearRegression

# create a Linear Regression object
lre = LinearRegression()
# fit the model using the feature "horsepower"
lre.fit(x_train[['horsepower']], y_train)
# calculate the R^2 on the test data
lre.score(x_test[['horsepower']], y_test)

lre.score(x_train[['horsepower']], y_train)

# Cross-Validation Score
from sklearn.model_selection import cross_val_score

# input the object, the feature ("horsepower"), and the target data (y_data).
# The parameter 'cv' determines the number of folds. In this case, it is 4.
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print(Rcross)

# calculate the average and standard deviation of our estimate:
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is", Rcross.std())

# use negative squared error as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'
-1 * cross_val_score(lre, x_data[['horsepower']], y_data, cv=4, scoring='neg_mean_squared_error')

# use the function 'cross_val_predict' to predict the output.
# The function splits up the data into the specified number of folds,
# with one fold for testing and the other folds are used for training.
from sklearn.model_selection import cross_val_predict

# input the object, the feature "horsepower", and the target data y_data. The parameter 'cv' determines the number of folds
yhat = cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4)
print(yhat[0:5])

# Overfitting, Underfitting and Model Selection
# create Multiple Linear Regression objects and train the model using 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
# Prediction using training data:
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
# Prediction using test data:
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# perform some model evaluation using our training and testing data separately
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
plt.show()
# test data
Title = 'Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test, yhat_test, "Actual Values (Test)", "Predicted Values (Test)", Title)
plt.show()

# Comparing Figure 1 and Figure 2, it is evident that the distribution of the test data in Figure 1 is much better at fitting the data.
# This difference in Figure 2 is apparent in the range of 5000 to 15,000. This is where the shape of the distribution is extremely different.
# Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset.
from sklearn.preprocessing import PolynomialFeatures

# overfitting
# use 55 percent of the data for training and the rest for testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
# perform a degree 5 polynomial transformation on the feature 'horsepower'
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
print(pr)

# create a Linear Regression model "poly" and train it
poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat = poly.predict(x_test_pr)

# use the function "PollyPlot" that we defined at the beginning of the lab to display the training data, testing data, and the predicted function.
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)
plt.show()

# R^2 of the training data and testing data
poly.score(x_train_pr, y_train)
poly.score(x_test_pr, y_test)

# Let's see how the R^2 changes on the test data for different order polynomials and then plot the results:
Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)

    x_train_pr = pr.fit_transform(x_train[['horsepower']])

    x_test_pr = pr.fit_transform(x_test[['horsepower']])

    lr.fit(x_train_pr, y_train)

    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
plt.show()


def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)
    plt.show()


f(3, 0.5)

# Ridge Regression
# degree two polynomial transformation on our data.
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(
    x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])
x_test_pr = pr.fit_transform(
    x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])

from sklearn.linear_model import Ridge

# create a Ridge regression object, setting the regularization parameter (alpha) to 0.1
RigeModel = Ridge(alpha=1)
# fit the model using the method fit
RigeModel.fit(x_train_pr, y_train)

yhat = RigeModel.predict(x_test_pr)
# compare the first four predicted samples to our test set:
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)

# select the value of alpha that minimizes the test error. To do so, we can use a for loop.
# We have also created a progress bar to see how many iterations we have completed so far
from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0, 1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha)
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)

    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

# plot out the value of R^2 for different alphas
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha, Rsqu_test, label='validation data  ')
plt.plot(Alpha, Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()

# The blue line represents the R^2 of the validation data, and the red line represents the R^2 of the training data.
# The x-axis represents the different values of Alpha.
# Here the model is built and tested on the same data, so the training and test data are the same.
# The red line in Figure 4 represents the R^2 of the training data. As alpha increases the R^2 decreases.
# Therefore, as alpha increases, the model performs worse on the training data
# The blue line represents the R^2 on the validation data. As the value for alpha increases, the R^2 increases and converges at a point.


# Grid Search
from sklearn.model_selection import GridSearchCV

# create a dictionary of parameter values
parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}]
# Create a Ridge regression object:
RR = Ridge()
# Create a ridge grid search object, In order to avoid a deprecation warning due to the iid parameter, we set the value of iid to "None"
Grid1 = GridSearchCV(RR, parameters1, cv=4, iid=None)

# Fit the model:
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
# The object finds the best parameter values on the validation data.
# We can obtain the estimator with the best parameters and assign it to the variable BestRR as follows:
BestRR = Grid1.best_estimator_
# test our model on the test data:
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)

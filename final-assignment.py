import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import rint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# load the file
file_name = 'source/kc_house_data_NaN.csv'
df = pd.read_csv(file_name)
print("5 rows: ")
print(df.head())
print("data types are:")
print(df.dtypes)
print('------------------------------------------------')

# Drop the columns "id" and "Unnamed: 0"
df.drop(["id", 'Unnamed: 0'], axis=1, inplace=True)
print(df.head())
print(df.describe())
print('------------------------------------------------')

# missing values for the columns  bedrooms and  bathrooms
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
# replace the missing values of the column 'bedrooms' with the mean of the column 'bedrooms', and the column 'bathrooms'
mean = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan, mean, inplace=True)
mean = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan, mean, inplace=True)
print('------------------------------------------------')
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
print('------------------------------------------------')

# count the number of houses with unique floor values
floors_counts = df['floors'].value_counts().to_frame()
print(floors_counts.head())
print('------------------------------------------------')

# determine whether houses with a waterfront view or without a waterfront view have more price outliers
sns.boxplot(x="waterfront", y="price", data=df)
plt.show()

# determine if the feature sqft_above is negatively or positively correlated with price
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="sqft_above", y="price", data=df)
plt.ylim(0, )
plt.show()

# use the Pandas method corr() to find the feature other than price that is most correlated with price.
df.corr()['price'].sort_values()

# Fit a linear regression model using the longitude feature 'long' and caculate the R^2.
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)

# Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2
X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)

#
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15",
            "sqft_above", "grade", "sqft_living"]
X = df[features]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)

# create a pipeline object to predict the 'price', fit the object using the features in the list features
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),
         ('model', LinearRegression())]
pipe = Pipeline(Input)
X = X.astype(float)
pipe.fit(X, Y)
pipe.score(X, Y)
ypipe = pipe.predict(X)

# split the data into training and testing sets:
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15",
            "sqft_above", "grade", "sqft_living"]
X = df[features]
Y = df['price']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
print("number of test samples:", x_test.shape[0])
print("number of training samples:", x_train.shape[0])
print('------------------------------------------------')

# Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data
RigeModel = Ridge(alpha=0.1)
RigeModel.fit(x_train, y_train)
test_score = RigeModel.score(x_test, y_test)
print(test_score)
print('-----------------------------------------------')

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RigeModel = Ridge(alpha=0.1)
RigeModel.fit(x_train_pr, y_train)
test_score = RigeModel.score(x_test_pr, y_test)
print(test_score)

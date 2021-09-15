# import pandas library
import pandas as pd
import numpy as np

# Import pandas library
import pandas as pd

# Read the online file by the URL provides above, and assign it to variable "df"
other_path = "source/auto.csv"
# no header for this dataset
df = pd.read_csv(other_path, header=None)
# df.head()   df.tail()

# create headers list
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
           "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg", "price"]
# add columns to the data frame
df.columns = headers

# replace the "?" symbol with NaN so the dropna() can remove the missing values
df1 = df.replace('?', np.NaN)

# drop missing values along the column "price"
df = df1.dropna(subset=["price"], axis=0)
df.head(20)

print(len(df.index))
len(df.columns)
df.info()

# save file
# df.to_csv("automobile.csv", index=False)

# check data types
print(df.dtypes)

# provide various summary statistics
df.describe()  # df.describe(include='all') provide a full summary statistics

# Where "column" is the name of the column, you can apply the method ".describe()" to get the statistics of those columns as follows:
# dataframe[[' column 1 ',column 2', 'column 3'] ].describe()

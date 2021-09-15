import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("source/m2_survey_data.csv")

# Plot the distribution curve for the column ConvertedComp.
ax1 = sns.distplot(df['ConvertedComp'], hist=False, color="b")
plt.show()

# Plot the histogram for the column `ConvertedComp`
ax2 = plt.hist(df['ConvertedComp'])
plt.show()

# What is the median of the column ConvertedComp?
df['ConvertedComp'].astype('float').median(axis=0)

# How many responders identified themselves only as a Man?
df['Gender'].value_counts()

# Find out the median ConvertedComp of responders identified themselves only as a Woman?
woman = df[df['Gender'] == 'Woman']
woman['ConvertedComp'].astype('float').median()

# Give the five number summary for the column Age?
df['Age'].describe()

# Plot a histogram of the column Age.
ax3 = plt.hist(df['Age'])
# plt.show()

# Find out if outliers exist in the column ConvertedComp using a box plot?
ax4 = sns.boxplot(data=df['ConvertedComp'])
# plt.show()

# Find out the Inter Quartile Range for the column ConvertedComp.
df["ConvertedComp"].describe()  # 75% - 25%

# Find out the upper and lower bounds.
Q1 = df["ConvertedComp"].quantile(0.25)
Q3 = df["ConvertedComp"].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# Identify how many outliers are there in the ConvertedComp column.
outliers = (df["ConvertedComp"] < (Q1 - 1.5 * IQR)) | (df["ConvertedComp"] > (Q3 + 1.5 * IQR))
outliers.value_counts()
print(outliers.value_counts())

# Create a new dataframe by removing the outliers from the ConvertedComp column.
df1 = df[~((df['ConvertedComp'] < (Q1 - 1.5 * IQR)) | (df['ConvertedComp'] > (Q3 + 1.5 * IQR)))]
print(df1.head())

# Find the correlation between Age and all other numerical columns.
df_num = df.select_dtypes(include=[np.float])
print(df_num)
df[['CompTotal', 'ConvertedComp', 'WorkWeekHrs', 'CodeRevHrs', 'Age']].corr()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

path = 'source/automobileEDA.csv'
df = pd.read_csv(path)
print(df.head())

# correlation
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()

# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0, )
df[["engine-size", "price"]].corr()  # 0.872335
plt.show()

sns.regplot(x="highway-mpg", y="price", data=df)
df[['highway-mpg', 'price']].corr()  # -0.704692
plt.show()

sns.regplot(x="peak-rpm", y="price", data=df)
df[['peak-rpm', 'price']].corr()  # -0.101616
plt.show()

# box plot
sns.boxplot(x="body-style", y="price", data=df)
plt.show()

# Value Counts
df['drive-wheels'].value_counts()
# convert the series to a dataframe
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
# rename the index to 'drive-wheels'
drive_wheels_counts.index.name = 'drive-wheels'

# group by
df_group_one = df[['drive-wheels', 'body-style', 'price']]
# calculate the average price for each of the different categories of data.
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()

# group multiple categories
# grouping results
df_gptest = df[['drive-wheels', 'body-style', 'price']]
grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
# leave the drive-wheels variable as the rows of the table, and pivot body-style to become the columns of the table:
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
grouped_pivot = grouped_pivot.fillna(0)  # fill missing values with 0

# find the average "price" of each car based on "body-style".
# grouping results
df_gptest2 = df[['body-style', 'price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'], as_index=False).mean()

# create pivot chart
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

# rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

# Correlation and Causation
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
# Since the p-value is  <  0.001, the correlation between wheel-base and price is statistically significant,
# although the linear relationship isn't extremely strong (~0.585).

# Width vs. Price
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

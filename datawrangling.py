import numpy as np
import pandas as pd

df = pd.read_csv("source/m1_survey_data.csv")

# deal with duplicate values
duplicate = df[df.duplicated()]
# duplicate = df[df.duplicated(subset = 'column name')]
print('before drop duplicate: ', len(duplicate.index))

print(duplicate.head())
df.drop_duplicates(inplace=True)
print('after drop duplicate: ', len(df[df.duplicated()].index))

# find the missing data
missing_data = df.isnull()  # true means missing data value
# Find the missing values for all columns.
df.isnull().sum()
# how many rows are missing in the column 'WorkLoc'
missing_workloc = df['WorkLoc'].isnull()
print(missing_workloc.value_counts())

# Identify the value that is most frequent (majority) in the WorkLoc column.
frequent_workloc = df['WorkLoc'].value_counts().idxmin()
# df['UndergradMajor'].value_counts().idxmin()
print('the most frequent work location is: ', frequent_workloc)
# replace missing value in WorkLoc column with frequent value
df['WorkLoc'].replace(np.nan, frequent_workloc, inplace=True)
print('after replace, the missing value: ', df['WorkLoc'].isnull().value_counts())
# df["WorkLoc"].fillna(value="Office",inplace=True)

# Normalizing data
# list out the various categories in the column 'CompFreq'
df['CompFreq'].unique()
print(df['CompFreq'].value_counts())

df["CompFreq"].replace(to_replace="Yearly", value=1, inplace=True)
df["CompFreq"].replace(to_replace="Monthly", value=12, inplace=True)
df["CompFreq"].replace(to_replace="Weekly", value=52, inplace=True)
df['NormalizedAnnualCompensation'] = df["CompTotal"] * df["CompFreq"]
print(df.head())
median = df['NormalizedAnnualCompensation'].astype('float').median(axis=0)
print('median is: ', median)

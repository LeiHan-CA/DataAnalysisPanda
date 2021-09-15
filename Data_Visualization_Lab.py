import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# open a database connection

conn = sqlite3.connect("source/m4_survey_data.sqlite")
# print how many rows are there in the table named 'master'
QUERY = """SELECT COUNT(*) FROM master"""  # tripe quota marks can easily be used for some '' string in the query
# the read_sql_query runs the sql query and returns the data as a dataframe
df = pd.read_sql_query(QUERY, conn)
print(df.head())

# list all tables
# print all the tables names in the database
QUERY = """
SELECT name as Table_Name FROM
sqlite_master WHERE
type = 'table'
"""
# the read_sql_query runs the sql query and returns the data as a dataframe
pd.read_sql_query(QUERY, conn)

# run a group by query
QUERY = """
SELECT Age,COUNT(*) as count
FROM master
group by age
order by age
"""
pd.read_sql_query(QUERY, conn)

# describe a table
table_name = 'master'  # the table you wish to describe

QUERY = """
SELECT sql FROM sqlite_master
WHERE name= '{}'
""".format(table_name)
df = pd.read_sql_query(QUERY, conn)
# print(df.iat[0,0])

# convert table to dataframe
QUERY = """
SELECT * 
FROM master
"""
df = pd.read_sql_query(QUERY, conn)

# Plot a histogram of ConvertedComp.
ax2 = plt.hist(df['ConvertedComp'])
plt.show()
# Plot a box plot of Age.
ax4 = sns.boxplot(data=df['Age'])

# Create a scatter plot of Age and WorkWeekHrs.
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df.Age, y=df.WorkWeekHrs, data=df)
plt.show()

# Create a bubble plot of WorkWeekHrs and CodeRevHrs, use Age column as bubble size.
plt.figure(figsize=(10, 5))
plt.scatter(x=df.WorkWeekHrs, y=df.CodeRevHrs, c='green', alpha=0.2, s=df.Age)
plt.show()

# pie chart
df2 = pd.read_sql_query('select * from DatabaseDesireNextYear', conn)

df2 = pd.read_sql_query(
    'select count(DatabaseDesireNextYear) as nextYear, DatabaseDesireNextYear from DatabaseDesireNextYear group by DatabaseDesireNextYear order by nextYear desc limit 5',
    conn)
df2.head()
# print(df2['nextYear'])
df2.groupby(['DatabaseDesireNextYear']).sum().plot(kind='pie', y='nextYear', startangle=90, figsize=(15, 10),
                                                   autopct='%1.1f%%')
plt.show()

# Create a stacked chart of median WorkWeekHrs and CodeRevHrs for the age group 30 to 35.
df_age = pd.read_sql_query('SELECT WorkWeekHrs, CodeRevHrs, Age FROM master', conn)
# group respondents by age and apply median() function
df_age = df_age.groupby('Age', axis=0).median()
# step 2: plot data
df_age[30:35].plot(kind='bar', figsize=(10, 6), stacked=True)
plt.xlabel('Age')  # add to x-label to the plot
plt.ylabel('Hours')  # add y-label to the plot
plt.title('Median Hours by Age')  # add title to the plot
plt.show()

# line chart
# step 1: get the data needed
df_comp = pd.read_sql_query('SELECT ConvertedComp, Age FROM master', conn)

# group respondents by age and apply median() function
df_comp = df_comp.groupby('Age', axis=0).median()

# step 2: plot data
df_comp[25:30].plot(kind='line', figsize=(10, 6), stacked=True)

plt.xlabel('Age')  # add to x-label to the plot
plt.ylabel('$')  # add y-label to the plot
plt.title('Median Compensation by Age')  # add title to the plot
plt.show()

# horizontal bar chart
df_main = pd.read_sql_query('SELECT MainBranch, count(MainBranch) as Count FROM master GROUP BY MainBranch', conn)
df_main.head()
df_main.plot(kind='barh', figsize=(10, 6))
plt.xlabel('Number of Respondents')  # add to x-label to the plot
plt.ylabel('Main Branch')  # add y-label to the plot
plt.title('Number of Respondents by Main Branch')  # add title to the plot
plt.show()

conn.close()

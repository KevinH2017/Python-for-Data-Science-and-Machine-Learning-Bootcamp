# Keras Regression Model Part 2

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to dataset
df = pd.read_csv("datasets/kc_house_data.csv")

# Drops 'id' column
df = df.drop('id', axis=1)

print(df.head())        # Prints first 5 rows of dataset
print(df.info())        # Prints summary of dataset

# Converts 'data' column to datetime object
df['date'] = pd.to_datetime(df['date'])
print(df['date'])

# Extracts month and year from 'date' column
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)

# Plots to visualize data
plt.figure('Box Plot for Month vs Price')
sns.boxplot(x='month', y='price', data=df)

plt.figure('Box Plot for Year vs Price')
sns.boxplot(x='year', y='price', data=df)

plt.figure('Mean Price per Month')
df.groupby('month').mean()['price'].plot()

plt.figure('Mean Price per Year')
df.groupby('year').mean()['price'].plot()

# Drops 'date' column
df = df.drop('date', axis=1)

# Drops 'zipcode' column
df = df.drop('zipcode',axis=1)

# Prints first 5 rows of modified dataset
print(df.head())

# Counts how many houses were renovated in each year
renovated_count = df['yr_renovated'].value_counts()
print(renovated_count)

# Counts how many houses have each value of 'sqft_basement'
sqft_basement_count = df['sqft_basement'].value_counts()
print(sqft_basement_count)

# Display plots
plt.show()
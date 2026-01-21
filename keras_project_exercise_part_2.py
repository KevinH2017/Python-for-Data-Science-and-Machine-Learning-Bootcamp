# Keras Project Exercise Part 2 - Data PreProcessing

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path to dataset 1
data_info = pd.read_csv('datasets/lending_club_info.csv', index_col='LoanStatNew')

def feat_info(col_name):
    '''Prints the description of a feature from the data_info dataframe.'''
    print(data_info.loc[col_name]["Description"])

# Path to dataset 2
df = pd.read_csv("datasets/lending_club_loan_two.csv")

# Prints length of dataframe
print("Length of dataframe:", len(df))

# Total count of missing data per columns
missing_data = df.isnull().sum()
print("Total count of missing data per columns:", missing_data)

# Percentage of missing data per columns
df_percentage = missing_data / len(df) * 100
print(df_percentage)

feat_info('emp_title')
feat_info('emp_length')

# Prints number of unique employement job titles
print(df['emp_title'].nunique())

# Prints the number of employment jobs for each unique title
print(df['emp_title'].value_counts())

# Drops emp_title from dataframe
df = df.drop('emp_title', axis=1) 

# Count plot of employment length
plt.figure("Count Plot of Employment Length", figsize=(10, 8))
plt.title("Count Plot of Employment Length")
# Orders emp_length by categories
emp_length_order = [ '< 1 year',
                    '1 year',
                    '2 years',
                    '3 years',
                    '4 years',
                    '5 years',
                    '6 years',
                    '7 years',
                    '8 years',
                    '9 years',
                    '10+ years']
sns.countplot(x='emp_length', data=df, order=emp_length_order)

# Count plot of employment length with hue separating Fully Paid vs Charged Off
plt.figure("Count Plot of Employment Length by Loan Status", figsize=(10, 4))
plt.title("Count Plot of Employment Length by Loan Status")
sns.countplot(x='emp_length', data=df, order=emp_length_order, hue='loan_status')

# Calculates the correlation between employment length and loan status
emp_co = df[df['loan_status'] == 'Charged Off'].groupby('emp_length').count()['loan_status']
emp_fp = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length').count()['loan_status']
emp_len = emp_co / (emp_co + emp_fp)
# Bar plot
plt.figure("Correlation between Employment Length and Loan Status", figsize=(10, 4))
plt.title("Correlation between Employment Length and Loan Status")
emp_len.plot(kind='bar')

# Drop emp_length column
df = df.drop('emp_length', axis=1)
print(df.isnull().sum())

# Prints the first 10 entries of purpose and title columns
print(df['purpose'].head(10))
print(df['title'].head(10))



# Show plots
plt.show()
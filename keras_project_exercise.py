# Keras Project Exercise

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import random

def feat_info(col_name):
    '''Prints the description of a feature from the data_info dataframe.'''
    print(data_info.loc[col_name]["Description"])

def fill_mort_acc(total_acc, mort_acc):
    '''Fills missing mort_acc values based on total_acc value'''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

# Path to dataset 1
data_info = pd.read_csv('datasets/lending_club_info.csv', index_col='LoanStatNew')

# Path to dataset 2
df = pd.read_csv("datasets/lending_club_loan_two.csv")

print(df.info())

# Count plot of loan status
plt.figure("Count Plot of Loan Status", figsize=(12, 6))
plt.title("Count Plot of Loan Status")
sns.countplot(x='loan_status', data=df)

# Histogram of Loan Amounts
plt.figure("Histogram of Loan Amount", figsize=(12, 6))
plt.title("Histogram of Loan Amount")
plt.hist(x=df['loan_amnt'], bins=50)

# Calculates and displays only numbers in the correlation matrix
df = pd.DataFrame(df)
num_df = df.select_dtypes(include=[np.number])
correlation_matrix = num_df.corr()
print(correlation_matrix)

# Heat map of correlation matrix
plt.figure("Heatmap of Correlation Matrix", figsize=(12, 6))
plt.title("Heatmap of Correlation Matrix")
sns.heatmap(correlation_matrix, annot=True)     # annot displays values inside boxes

# Scatter plot of installments vs loan amounts
plt.figure("Installment vs Loan Amount", figsize=(12, 6))
plt.title("Installment vs Loan Amount")
plt.xlabel("Installment")
plt.ylabel("Loan Amount")
plt.scatter(df['installment'], df['loan_amnt'], edgecolors='white')

# Box plot of loan status vs loan amount
plt.figure("Loan Status vs Loan Amount")
plt.title("Loan Status vs Loan Amount")
sns.boxplot(x='loan_status', y='loan_amnt', data=df)

# Statistics of numerical columns grouped by loan status
loan_status_stats = (df.set_index("loan_status")
                        .select_dtypes(np.number)
                        .stack()
                        .groupby(level=0)
                        .describe())
print(loan_status_stats)

# Unique values in 'grade' and 'sub_grade' columns
grades = df['grade'].unique()
subgrades = df['sub_grade'].unique()
print(sorted(grades))
print(sorted(subgrades))

subgrade_order = sorted(df['sub_grade'].unique())

# Count plot of loan grades
plt.figure("Count Plot of Loan Grades", figsize=(12, 6))
plt.title("Count Plot of Loan Grades")
sns.countplot(x='grade', data=df, hue='loan_status')

# Count plot of sub loan grades
plt.figure("Count Plot of Loan Status by Sub Grades", figsize=(12, 4))
plt.title("Count Plot of Loan Status by Sub Grades")
sns.countplot(x='sub_grade', data=df, order=subgrade_order)

# Count plot of loan status by grades
plt.figure("Count Plot of Loan Status by Grade", figsize=(12, 4))
plt.title("Count Plot of Loan Status by Grade")
sns.countplot(x='sub_grade', data=df, order=subgrade_order, hue='loan_status')

# Count plot of sub grades of grade categories F and G
f_and_g = df[(df['grade']=='F') | (df['grade']=='G')]
subgrade_order = sorted(f_and_g['sub_grade'].unique())
plt.figure("Count Plot of Loan Status Grades F and G", figsize=(12, 4))
plt.title("Count Plot of Loan Status Grades F and G")
sns.countplot(x='sub_grade', data=f_and_g, order=subgrade_order, hue='loan_status')

# Creates a new column 'loan_repaid' mapping loan status to 'Fully Paid' as 1 and 'Charged Off' as 0
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1, 'Charged Off':0})

# Bar plot to show correlation between 'Fully Paid' and 'Charged Off' loans
plt.figure("Correlation of Loan Repaid")
plt.title("Correlation of Loan Repaid")
df.corr(numeric_only=True)['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')

# Prints length of dataframe
print("Length of dataframe:", len(df))

# Total count of missing data per columns
missing_data = df.isnull().sum()
print("Total count of missing data per columns:", missing_data)

# Percentage of missing data per columns
df_percentage = missing_data / len(df) * 100
print(df_percentage)

# Information about emp_title and emp_length
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
feat_info('purpose')
print(df['purpose'].head(10))
feat_info('title')
print(df['title'].head(10))

# Drops title column
df = df.drop('title', axis=1)

# Information about mort_acc
feat_info('mort_acc')
print(df['mort_acc'].value_counts())
# Correlation between mort_acc and other columns
df.corr(numeric_only=True)['mort_acc'].sort_values()

# Groups dataframe by total_acc and finds mean of mort_acc per total_acc entry
total_acc_avg = df.groupby('total_acc').mean(numeric_only=True)['mort_acc']
print(total_acc_avg)

# Fills in missing mort_acc values with average corresponding mort_acc for each total_acc value
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

# Drop null values
df = df.dropna()

# Information about mort_acc
feat_info('mort_acc')
print(df['mort_acc'].value_counts())
# Correlation between mort_acc and other columns
df.corr(numeric_only=True)['mort_acc'].sort_values()

# Groups dataframe by total_acc and finds mean of mort_acc per total_acc entry
total_acc_avg = df.groupby('total_acc').mean(numeric_only=True)['mort_acc']
print(total_acc_avg)

# Fills in missing mort_acc values with average corresponding mort_acc for each total_acc value
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

# Drop null values
df = df.dropna()
print(df.isnull().sum())

# Map strings '36 months' and '60 months' to respective integers
df['term'] = df['term'].apply(lambda term: int(term[:3]))
0
df = df.drop('grade', axis=1)

# Converts categorical values into dummy variables
dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
# Concats dropped 'subgrade' column, replaced with dummy variables
df = pd.concat([df.drop('sub_grade', axis=1), dummies], axis=1)

dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose']], drop_first=True)
df = pd.concat([df.drop(['verification_status', 'application_type','initial_list_status','purpose'], axis=1), dummies], axis=1)

# Combines NONE and ANY values into OTHER
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = pd.concat([df.drop('home_ownership', axis=1), dummies], axis=1)

# Creates 'zip_code' columns from 'address'
df['zip_code'] = df['address'].apply(lambda address: address[-5:])
dummies = pd.get_dummies(df['zip_code'], drop_first=True)
df = pd.concat([df.drop('zip_code', axis=1), dummies], axis=1)

df = df.drop('address', axis=1)
df = df.drop('issue_d', axis=1)

# Convert 'earliest_cr_line' to year date
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
print(df['earliest_cr_line'])

df = df.drop('loan_status', axis=1)

# Set X and y variables for train test split
X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Normalize data and into a given range and transform it for processing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building Sequential model
model = Sequential()

model.add(Dense(units=78, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=39, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=19, activation='relu'))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Configures model to be compiled and used for training
model.compile(loss='binary_crossentropy', optimizer='adam')

# Model training settings
model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test))

# Saves model to file
model.save('keras_project_model.keras')
# Loads model from file
# model = load_model('keras_project_model.keras')

# Plots training loss from model
losses = pd.DataFrame(model.history.history)
losses.plot()

# Creates predictions from X_test and displys classification report
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)       # Fixes ValueError:Classification metrics can't handle a mix of binary and continuous targets
print(classification_report(y_test, predictions))

print("Counts of loans that have been repaid or not repaid:")
print(df['loan_repaid'].value_counts())
print("Rough percentage of model accuracy:")
print(317696 / len(df))

# Uncomment to display plots
# plt.show()
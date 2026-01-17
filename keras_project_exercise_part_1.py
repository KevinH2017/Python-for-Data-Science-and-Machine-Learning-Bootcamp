# Keras Project Exercise Part 1 - Exploratory Data Analysis

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path to dataset 1
data_info = pd.read_csv('datasets/lending_club_info.csv', index_col='LoanStatNew')

print(data_info.loc['revol_util']['Description'])

def feat_info(col_name):
    print(data_info.loc[col_name]["Description"])

print(feat_info("mort_acc"))

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

# Show plots
plt.show()
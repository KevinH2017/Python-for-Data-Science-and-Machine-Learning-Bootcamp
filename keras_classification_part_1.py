# Keras Classification Part 1

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to dataset
df = pd.read_csv('datasets/cancer_classification.csv')

# Prints information about the dataset
print(df.info())
print(df.describe().transpose())

# Count plot of benign and malignant tumors
plt.figure("Count Plot of Benign(0) and Malignant(1) Tumors")
plt.title("Count Plot of Benign(0) and Malignant(1) Tumors")
sns.countplot(x='benign_0__mal_1', data=df)

# Heat map of cancer classifications
plt.figure("Heatmap of Cancer Classification Dataset", figsize=(12, 10))
plt.title("Heatmap of Cancer Classification Dataset")
sns.heatmap(df.corr())

# Prints correlation values with respect to each benign and malignant columns
print(df.corr()['benign_0__mal_1'].sort_values())

# Bar plot of correlation values with respect to benign and malignant columns
plt.figure("Bar Plot of Correlation Values with respect to Benign(0) and Malignant(1) Columns")
plt.title("Bar Plot of Correlation Values with respect to Benign(0) and Malignant(1) Columns")
df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')

# Bar plot of correlation values excluding the target column
plt.figure("Bar Plot of Correlation Values excluding Target Column")
plt.title("Bar Plot of Correlation Values excluding Target Column")
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

# Show plots
plt.show()
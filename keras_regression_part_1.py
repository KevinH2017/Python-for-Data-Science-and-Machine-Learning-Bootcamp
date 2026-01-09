# Keras Regression Model Part 1

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to dataset
df = pd.read_csv("datasets/kc_house_data.csv")
print(df.describe().transpose())

# Distribution plot for prices
plt.figure("Distribution Plot for Prices")
sns.displot(df['price'])

# Count plot for bedrooms
plt.figure("Count Plot for Bedrooms")
sns.countplot(data=df, x='bedrooms')

# Scatter plot for price vs sqft_living
plt.figure("Scatter Plot for Price vs Sqft Living")
sns.scatterplot(x='price', y='sqft_living', data=df)

# Box plot for bedrooms vs price
plt.figure("Box Plot for Bedrooms vs Price")
sns.boxplot(x='bedrooms', y='price', data=df)

# Scatter plot for price vs longitude
plt.figure("Scatter Plot for Price vs Longitude")
sns.scatterplot(x='price', y='long', data=df)

# Scatter plot for price vs latitude
plt.figure("Scatter Plot for Price vs Latitude")
sns.scatterplot(x='price', y='lat', data=df)

# Removes the top 1% of prices to reduce outlier values
non_top_1_percent = df.sort_values('price', ascending=False).iloc[216:]

# Scatter plot for longitude vs latitude
plt.figure("Scatter Plot for Longitude vs Latitude")
sns.scatterplot(x='long', y='lat', data=non_top_1_percent, 
                edgecolor=None, alpha=0.2, palette='RdYlGn', hue='price')

# Box plot for waterfront vs price
plt.figure("Box Plot for Waterfront vs Price")
sns.boxplot(x='waterfront', y='price', data=df)

# Show plots
plt.show()
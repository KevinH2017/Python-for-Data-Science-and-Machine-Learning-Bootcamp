# Keras Regression Model Part 3

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from keras.models import Sequential
from keras.layers import Dense

# Path to dataset
df = pd.read_csv("datasets/kc_house_data.csv")

# Drops 'id' column
df = df.drop('id', axis=1)

# Converts 'data' column to datetime object
df['date'] = pd.to_datetime(df['date'])

# Extracts month and year from 'date' column
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)

# Drops 'date' column
df = df.drop('date', axis=1)

# Drops 'zipcode' column
df = df.drop('zipcode',axis=1)

# Drops 'price' column
X = df.drop('price', axis=1)
y = df['price']

# Splits dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

# Scales features using MinMaxScaler to a range of 0 or 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creates dense neural network and adds layers with 19 neurons each
model = Sequential()
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

# Output layer
model.add(Dense(1))

# Configures model to be compiled and used for training
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, 
    validation_data=(X_test, y_test), 
    batch_size=128, epochs=400)

losses = pd.DataFrame(model.history.history)
losses.plot()

# Evaluates model performance by check loss on test data
predictions = model.predict(X_test)
mse_result = np.sqrt(mean_squared_error(y_test, predictions))
mae_result = mean_absolute_error(y_test, predictions)
variances_score = explained_variance_score(y_test,predictions)

print("MSE:", mse_result)
print("MAE:", mae_result)
print("Explained Variance Score:", variances_score)

# Visualizes predictions vs actual prices
price_mean = df['price'].mean()
price_median = df['price'].median()

print("Price Mean:", price_mean)
print("Price Median:", price_median)

plt.figure(figsize=(12, 6))

# Our predictions
plt.title("Predicted vs Actual Prices")
plt.scatter(y_test, predictions)

# Perfect predictions
plt.title("Predicted vs Actual Prices")
plt.plot(y_test, y_test, 'r')

errors = y_test.values.reshape(6480, 1) - predictions
plt.title("Distribution of Errors")
sns.displot(errors)

# Predicts price for a single house
single_house = df.drop('price', axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 19))

# Predicts the cost of a single house
model.predict(single_house)
print(df.iloc[0])

# Show plots
plt.show()
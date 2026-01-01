import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense

tf.__version__

# Path to dataset
df = pd.read_csv("datasets/fake_reg.csv")
print(df.head())

# Features
X = df[["feature1", "feature2"]].values

# Label
y = df["price"].values

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Transforms the dataset to a fixed range of [0,1]
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# # Creates dense neural networks with a specific amount of neurons in each layer as a list passed to Sequential()
# model = Sequential([Dense(4, activation='relu'),
#                     Dense(2, activation='reul'),
#                     Dense(1)])

# Creates dense neural networks with a specific amount of neurons in each layer as separate lines
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# Configurations for model training
model.compile(optimizer='rmsprop', loss='mse')

# Trains the model over iterations of training dataset
model.fit(x=X_train, y=y_train, epochs=250)

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

# Display graph
plt.show()
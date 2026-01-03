# Keras Syntax for Dense Neural Networks

# Import libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import seaborn as sns
from keras.models import Sequential, load_model
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

# Creates dense neural networks and adds a number of neurons in each layer as a list passed to Sequential()
# model = Sequential([Dense(4, activation='relu'),
#                     Dense(4, activation='relu'),
#                     Dense(4, activation='relu'), 
#                     Dense(1)])

# Creates dense neural networks and adds a number of neurons in each layer as separate lines
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

# Uses training set and test set to evaluate model performance
model.evaluate(X_test, y_test, verbose=0)
model.evaluate(X_train, y_train, verbose=0)

# Plotting training and test sets loss
test_predictions = model.predict(X_test)

# Converts predictions into a series and dataframe for better processing
test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.DataFrame(y_test, columns=['Test True Y'])

# Combines true test labels and model predictions into a single dataframe
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['Test True Y', 'Model Predictions']

# Calculates error metrics to evaluate model accuracy
# Mean Absolute Error averages the absolute value of the errors
mae = mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions'])

# Mean Squared Error averages the squared value of the errors
mse = mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])

# Root Mean Squared Error is the square root of the mean squared error
rmse = root_mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# # Scatter plot to show how accurate the model predictions are
# sns.scatterplot(x='Test True Y', y='Model Predictions', data=pred_df)
# # Display graph
# plt.show()

# Predicting value using brand new data
# [[Feature 1, Feature 2]]
new_gem = [[998, 1000]]
# Scales the new data
new_gem = scaler.transform(new_gem)

model.predict(new_gem)

# Saving and loading a model
model.save('my_model.h5')       # Creates a HDF5 file

# Loads model from HDF5 file
later_model = load_model('my_model.h5')

# Uses loaded model to make prediction
later_model.predict(new_gem)
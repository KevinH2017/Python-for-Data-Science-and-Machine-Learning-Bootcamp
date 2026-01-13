# Keras Overfitting Example

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

# Path to dataset
df = pd.read_csv('datasets/cancer_classification.csv')

# Splitting dataset into training and testing sets
X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101)

# Scaling features using MinMaxScaler to a range of 0 or 1
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)

# Creates dense neural network and adds layers with 30 and 15 neurons
model = Sequential()
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=15, activation='relu'))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Configures model to be compiled and used for training
model.compile(loss='binary_crossentropy', optimizer='adam')

# Example of overfitting a model with too many epochs
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1)

print(model.history.history)

# Plotting training and validation loss
losses = pd.DataFrame(model.history.history)
losses.plot()

# Show plots
plt.show()
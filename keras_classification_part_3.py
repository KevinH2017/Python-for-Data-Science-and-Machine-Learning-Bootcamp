# Keras Classification Part 3

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

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

# Creates dense neural network and adds layers with 30 and 15 neurons
model = Sequential()
model.add(Dense(units=30, activation='relu'))

# Adding dropout layer to reduce overfitting
# Dropout layer randomly drops 50% of the neurons during training
model.add(Dropout(0.5))

model.add(Dense(units=15, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Configures model to be compiled and used for training
model.compile(loss='binary_crossentropy', optimizer='adam')

# Stops training when validation loss stops improving to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=25)

# Training the model with early stopping
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop])

print(model.history.history)

# Plotting training and validation loss
losses = pd.DataFrame(model.history.history)
# Plots with both Early Stopping and Dropout
losses.plot()

# Generating predictions from the test set
predictions = model.predict(X_test)

# Classification report and confusion matrix using test set and predictions
print(classification_report(y_test, predictions.round()))
print(confusion_matrix(y_test, predictions.round()))

# Show plots
plt.show()
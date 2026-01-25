# Tensorboard Practice

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime

# Path to dataset
df = pd.read_csv('datasets/cancer_classification.csv')

# Drop 'benign_0__mal_1' column for X
X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101)

# Scale the data
scaler = MinMaxScaler()
scaler.fit(X_train)

# Scale training and test data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Early stopping to reduce overfitting 
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=25)

# Directory to save logs with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d--%H%M")
log_directory = 'logs\\fit'
log_directory = log_directory + '\\' + timestamp

board = TensorBoard(
    log_dir=log_directory,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1)

# Building model and layers
model = Sequential()
model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=15, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# Training the model with early stopping and tensorboard
model.fit(
    x=X_train, 
    y=y_train, 
    epochs=600,
    validation_data=(X_test, y_test), 
    verbose=1,
    callbacks=[early_stop, board])

# Use below command in command line to view tensorboard stats
# tensorboard --logdir logs\fit 
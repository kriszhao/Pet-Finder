import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                    sep=';')

# Read in red wine data 
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

# Print info on white wine
print(white.info())

# Print info on red wine
print(red.info())

# First rows of `red`
print(red.head())

# Last rows of `white`
print(white.tail())

# Take a sample of 5 rows of `red`
print(red.sample(5))

# Describe `white`
print(white.describe())

# Double check for null values in `red`
print(pd.isnull(red))

np.random.seed(570)

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])

# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

# Specify the data
# X = wines.ix[:, 0:11]

# Specify the target labels and flatten the array
# y = np.ravel(wines.type)

# Isolate target labels
y = wines.quality

# Isolate data
X = wines.drop('quality', axis=1)

# Scale the data with `StandardScaler`
X = StandardScaler().fit_transform(X)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define the scaler
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# Model output shape
print(model.output_shape)

# Model summary
print(model.summary())

# Model config
print(model.get_config())

# List all weight tensors
print(model.get_weights())

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=500, batch_size=1, verbose=1)

y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test, verbose=1)

print(score)

# # Import the modules from `sklearn.metrics`
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
#
# # Confusion matrix
# confusion_matrix(y_test, y_pred)
#
# print(precision_score(y_test, y_pred))
#
# # Recall
# print(recall_score(y_test, y_pred))
#
# # F1 score
# print(f1_score(y_test,y_pred))
#
# # Cohen's kappa
# print(cohen_kappa_score(y_test, y_pred))

# ----------------

# # Isolate target labels
# Y = wines.quality
#
# # Isolate data
# X = wines.drop('quality', axis=1)
#
# # Scale the data with `StandardScaler`
# X = StandardScaler().fit_transform(X)
#
# from keras.models import Sequential
#
# # Import `Dense` from `keras.layers`
# from keras.layers import Dense
#
# # Initialize the model
# model = Sequential()
#
# # Add input layer
# model.add(Dense(64, input_dim=12, activation='relu'))
#
# # Add output layer
# model.add(Dense(1))
#
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
#
# seed = 7
# np.random.seed(seed)
#
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#
# # model = Sequential()
# # model.add(Dense(128, input_dim=12, activation='relu'))
# # model.add(Dense(1))
# #
# # from keras.optimizers import RMSprop
# # rmsprop = RMSprop(lr=0.0001)
# # model.compile(optimizer=rmsprop, loss='mse', metrics=['mae'])
# #
# # from keras.optimizers import SGD, RMSprop
# # sgd=SGD(lr=0.1)
# # model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
#
# for train, test in kfold.split(X, Y):
#     model = Sequential()
#     model.add(Dense(64, input_dim=12, activation='relu'))
#     model.add(Dense(1))
#     model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#     model.fit(X[train], Y[train], epochs=10, verbose=1)
#     y_pred = model.predict(X[test])
#
#     mse_value, mae_value = model.evaluate(X[test], Y[test], verbose=0)
#
#     print(mse_value)
#     print(mae_value)
#
#     from sklearn.metrics import r2_score
#
#     r2_score(Y[test], y_pred)

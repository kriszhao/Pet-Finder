# libraries
import warnings

import keras.callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Reshape
from keras.layers import Embedding, Dropout, Concatenate, Input
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

img_size = 256
batch_size = 16
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2


# Measure of success
def kappa(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def preproc(X_train, X_test, embed_cols, num_cols):
    input_list_train = []
    input_list_test = []
    m = MinMaxScaler()

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}

        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i

        m.fit(X_train[c].map(val_map).values.reshape(-1, 1))
        input_list_train.append(m.transform(X_train[c].map(val_map).values.reshape(-1, 1)))

        m.fit(X_test[c].map(val_map).fillna(0).values.reshape(-1, 1))
        input_list_test.append(m.transform(X_test[c].map(val_map).fillna(0).values.reshape(-1, 1)))

    # the numerical columns
    m.fit(X_train[num_cols].values)
    input_list_train.append(m.transform(X_train[num_cols].values))

    m.fit(X_test[num_cols].values)
    input_list_test.append(m.transform(X_test[num_cols].values))

    # img data
    input_list_train.append(X_train.iloc[:, 19:].as_matrix())
    input_list_test.append(X_test.iloc[:, 19:].as_matrix())

    return input_list_train, input_list_test


# Creating a Embedding model for categorical variables using the fast.ai approach
def createModel(data, categorical_vars, numerical_vars):
    embeddings = []
    inputs = []

    for categorical_var in categorical_vars:
        i = Input(shape=(1,))
        no_of_unique_cat = data[categorical_var].nunique()
        embedding_size = min(np.ceil(no_of_unique_cat / 2), 50)
        embedding_size = int(embedding_size)
        vocab = no_of_unique_cat + 1
        embedding = Embedding(vocab, embedding_size, input_length=1)(i)
        embedding = Reshape(target_shape=(embedding_size,))(embedding)
        embeddings.append(embedding)
        inputs.append(i)

    input_numeric = Input(shape=(len(numerical_vars),))
    embedding_numeric = Dense(16)(input_numeric)

    inputs.append(input_numeric)
    embeddings.append(embedding_numeric)

    x = Concatenate()(embeddings)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.25)(x)
    x = Dense(30, activation='relu')(x)
    x = Dropout(.25)(x)

    # hardcoded for now
    image_input = Input(shape=(256,))
    inputs.append(image_input)

    y = Dense(80, activation='relu')(image_input)
    y = Dense(40, activation='relu')(y)
    y = Dropout(.25)(y)

    z = Concatenate()([x, y])

    z = Dense(20, activation='relu')(z)

    output = Dense(5, activation='sigmoid')(z)

    model = Model(inputs, output)
    model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='adam')
    return model


if __name__ == '__main__':
    breeds = pd.read_csv('../all/breed_labels.csv')
    colors = pd.read_csv('../all/color_labels.csv')
    states = pd.read_csv('../all/state_labels.csv')
    train = pd.read_csv('../all/train.csv')
    test = pd.read_csv('../all/test/test.csv')

    pet_ids = train['PetID'].values
    n_batches = len(pet_ids) // batch_size + 1

    train_feats = pd.read_csv('../all/train_img_features.csv')
    test_feats = pd.read_csv('../all/test/test_img_features.csv')

    print(train_feats.shape)
    print(test_feats.shape)

    train = train.merge(train_feats, left_on='PetID', right_on='Unnamed: 0', how='outer')

    train_label = train.AdoptionSpeed
    # We drop name because it creates a huge embedding vector and we know that name is not very useful anyway
    train = train.drop(['AdoptionSpeed', 'Name', 'Description', 'PetID', 'Unnamed: 0', 'RescuerID'], axis=1)

    # create train and test set
    X_train, X_test, y_train, y_test = train_test_split(train, train_label, test_size=TEST_SPLIT, random_state=9)

    # Turn labels into n dimensional vectors for loss calculation
    y_train = to_categorical(y_train, num_classes=None)
    y_test = to_categorical(y_test, num_classes=None)

    categorical_vars = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength',
                        'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']
    numerical_vars = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']

    warnings.filterwarnings('ignore')
    X_train, X_test = preproc(X_train, X_test, categorical_vars, numerical_vars)

    model = createModel(train, categorical_vars, numerical_vars)
    print(model.summary())

    filepath = '../checkpoints/weights_image_categorical.hdf6'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                 mode='max')
    early_stopped = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10, verbose=0, mode='max')
    callbacks_list = [checkpoint]

    hist = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=VALIDATION_SPLIT,
                     shuffle=True, callbacks=callbacks_list)

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.plot()

    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    plt.plot()

    model.evaluate(X_test, y_test)

    test_pred = model.predict(X_test)
    print(kappa(y_test, test_pred))

    train_pred = model.predict(X_train)
    print(kappa(y_train, train_pred))

    model.load_weights('../checkpoints/weights_image_categorical.hdf6')

    test_pred = model.predict(X_test)
    print(kappa(y_test, test_pred))

    train_pred = model.predict(X_train)
    print(kappa(y_train, train_pred))

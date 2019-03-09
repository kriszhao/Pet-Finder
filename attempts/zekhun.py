import re

import matplotlib.pyplot as plt

import pandas as pd
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import np
from keras.optimizers import RMSprop
from keras.utils import np_utils
from numpy import array
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(1337)  # for reproducibility


def extract_max(input):
    return list(map(int, re.findall('\d+', input)))


def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


if __name__ == '__main__':
    train_df = pd.read_csv("../all/train.csv")
    test_df = pd.read_csv("../all/test/test.csv")

    cat_cols = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
                'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'State', 'VideoAmt',
                'PhotoAmt']
    num_cols = ['Fee']
    text_cols = ['Description']

    print('Scaling num_cols')

    for col in num_cols:
        print('scaling {}'.format(col))
        col_mean = train_df[col].mean()
        train_df[col].fillna(col_mean, inplace=True)
        test_df[col].fillna(col_mean, inplace=True)
        scaler = StandardScaler()
        train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1))
        test_df[col] = scaler.transform(test_df[col].values.reshape(-1, 1))

    tr_df, val_df = train_test_split(train_df, test_size=0.5, random_state=4)

    y_train = np_utils.to_categorical(tr_df['AdoptionSpeed'], num_classes=5)
    y_valid = np_utils.to_categorical(val_df['AdoptionSpeed'], num_classes=5)

    cs = StandardScaler()
    trainContinuous = cs.fit_transform(tr_df[cat_cols])
    trainContinuous2 = cs.fit_transform(val_df[cat_cols])
    trainContinuous3 = cs.fit_transform(test_df[cat_cols])
    x_train = np.hstack([trainContinuous])
    x_valid = np.hstack([trainContinuous2])
    x_test = np.hstack([trainContinuous3])

    model = Sequential([
        Dense(64, input_dim=18),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(16),
        Activation('relu'),
        Dense(5),
        Activation('sigmoid'),
    ])

    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # We add metrics to get more results you want to see
    # categorical_crossentropy--mse
    model.compile(optimizer=rmsprop,
                  loss='mse',
                  metrics=['accuracy'])

    filepath = "../checkpoints/weights_image_categorical.hdf6"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    earlystopped = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10, verbose=0, mode='max')
    callbacks_list = [checkpoint]

    for i in range(10):
        x_train1, x_train2, y_train1, y_train2 = train_test_split(x_train, y_train, test_size=0.2, random_state=i * 15)
        history = model.fit(x_train1, y_train1, validation_data=(x_train2, y_train2), epochs=100, batch_size=1000,
                            shuffle=True, callbacks=callbacks_list)

    model.load_weights('../checkpoints/bestForNow.hdf6')

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss = model.predict(x_train, batch_size=1000)
    ans = [0 for x in range(len(x_train))]

    for i in range(len(x_train)):
        index_of_maximum = np.where(loss[i] == loss[i].max())
        ans[i] = extract_max(str(index_of_maximum))

    ans = array(ans)

    y_train = tr_df['AdoptionSpeed'].values

    loss1 = model.predict(x_valid, batch_size=1000)
    ans1 = [0 for x in range(len(x_valid))]

    for i in range(len(x_valid)):
        index_of_maximum = np.where(loss1[i] == loss1[i].max())
        ans1[i] = extract_max(str(index_of_maximum))

    ans1 = array(ans1)

    y_valid = val_df['AdoptionSpeed'].values

    y_train_pred = ans
    y_valid_pred = ans1
    avg_train_kappa = 0
    avg_valid_kappa = 0
    avg_train_kappa += kappa(y_train_pred, y_train)
    avg_valid_kappa += kappa(y_valid_pred, y_valid)
    print("\navg train kappa:", avg_train_kappa, )
    print("\navg valid kappa:", avg_valid_kappa, )

    f = plt.figure(figsize=(10, 3))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid'], loc='upper left')

    ax2.plot(history.history['acc'])
    ax2.plot(history.history['val_acc'])
    ax2.legend(['acc', 'val_acc'])
    ax2.plot()

import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.layers import Dropout, Dense, np
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

# CONSTANTS
HIDDEN_UNITS = [100, 50, 25, 12]
LABEL = 'AdoptionSpeed'
TRAINING_TEST_SPLIT = 0.2
RANDOM_NUMBER_SEED = 42
N_CLASSES = 5
EPOCHS = 100
TRAIN_BATCH_SIZE = 10
TRAIN_FILENAME = 'weights.best.hdf5'

np.random.seed(RANDOM_NUMBER_SEED)


def prepare_data(data):
    pet_id = data.PetID

    # Remove unused features
    data.drop(['RescuerID', 'Description', 'PetID', 'State'], axis=1, inplace=True)

    # Apply binning to ages
    data['Age'] = pd.cut(data['Age'], [-1, 2, 3, 6, 255], labels=[0, 1, 2, 3])

    # Apply binning to fee
    data['Fee'] = pd.cut(data['Fee'], [-1, 50, 100, 200, 3000], labels=[0, 1, 2, 3])

    # Apply binning to photo amount
    data['PhotoAmt'] = pd.cut(data['PhotoAmt'], [-1, 1, 5, 10, 100], labels=[0, 1, 2, 3])

    # Apply binning to video amount
    data['VideoAmt'] = pd.cut(data['VideoAmt'], [-1, 1, 100], labels=[0, 1])

    # Replace names with 1 is present, 0 if not present
    data.loc[data['Name'].notnull(), 'Name'] = 1
    data.loc[data['Name'].isnull(), 'Name'] = 0

    # Fill missing continuous data
    data_continuous = data.select_dtypes(exclude=['object'])
    data_continuous.fillna(0, inplace=True)

    # Fill missing string data
    data_categorical = data.select_dtypes(include=['object'])
    data_categorical.fillna('NONE', inplace=True)

    final_data = data_continuous.merge(data_categorical, left_index=True, right_index=True)

    return final_data, data_categorical, data_continuous, pet_id, data.shape[1]


def create_mlp(input_dim, output_dim, dropout=0.3, arch=None):
    # Default mlp architecture
    arch = arch if arch else [64, 32, 32, 16]

    # Setup densely connected NN architecture (MLP)
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(input_dim,)))

    for output in arch:
        model.add(Dense(output, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropout))

    model.add(Dense(output_dim, activation='sigmoid'))

    # Compile model and save architecture to disk
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # Import and split
    train, train_categorical, train_continuous, train_pet_id, training_dimension = prepare_data(
        pd.read_csv('../all/train.csv'))
    test, test_categorical, test_continuous, test_pet_id, _ = prepare_data(pd.read_csv('../all/test/test.csv'))

    # Remove the outliers
    clf = IsolationForest(max_samples=100, random_state=RANDOM_NUMBER_SEED)
    clf.fit(train_continuous)
    y_no_outliers = clf.predict(train_continuous)
    y_no_outliers = pd.DataFrame(y_no_outliers, columns=['Top'])

    train_continuous = train_continuous.iloc[y_no_outliers[y_no_outliers['Top'] == 1].index.values]
    train_continuous.reset_index(drop=True, inplace=True)

    train_categorical = train_categorical.iloc[y_no_outliers[y_no_outliers['Top'] == 1].index.values]
    train_categorical.reset_index(drop=True, inplace=True)

    train = train.iloc[y_no_outliers[y_no_outliers['Top'] == 1].index.values]
    train.reset_index(drop=True, inplace=True)

    # Extract columns
    columns = list(train_continuous.columns)

    features_continuous = list(train_continuous.columns)
    features_continuous.remove(LABEL)

    features_categorical = list(train_categorical.columns)

    # Extract matrices
    matrix_train = np.matrix(train_continuous)
    matrix_test = np.matrix(test_continuous)
    matrix_test_no_label = np.matrix(train_continuous.drop(LABEL, axis=1))
    matrix_y = np.array(train.AdoptionSpeed)

    # Scale data
    y_scaler = MinMaxScaler()
    y_scaler.fit(matrix_y.reshape(matrix_y.shape[0], 1))

    train_scaler = MinMaxScaler()
    train_scaler.fit(matrix_train)

    test_scaler = MinMaxScaler()
    test_scaler.fit(matrix_test_no_label)

    matrix_train_scaled = pd.DataFrame(train_scaler.transform(matrix_train), columns=columns)
    test_matrix_scaled = pd.DataFrame(test_scaler.transform(matrix_test), columns=features_continuous)

    train[columns] = pd.DataFrame(train_scaler.transform(matrix_train), columns=columns)
    test[features_continuous] = test_matrix_scaled

    # Extract continuous and categorical features
    engineered_features = []

    for continuous_feature in features_continuous:
        engineered_features.append(tf.contrib.layers.real_valued_column(continuous_feature))

    for categorical_feature in features_categorical:
        sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(categorical_feature, hash_bucket_size=1000)

        engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column,
                                                                      dimension=16,
                                                                      combiner='sum'))

    # Split training set data between train and test
    x_train, x_test, y_train, y_test = train_test_split(train[features_continuous + features_categorical],
                                                        train[LABEL],
                                                        test_size=TRAINING_TEST_SPLIT,
                                                        random_state=RANDOM_NUMBER_SEED)

    # Convert back to DataFrame
    y_train = pd.DataFrame(y_train, columns=[LABEL])
    x_train = pd.DataFrame(x_train, columns=features_continuous + features_categorical) \
        .merge(y_train, left_index=True, right_index=True)

    y_test = pd.DataFrame(y_test, columns=[LABEL])
    x_test = pd.DataFrame(x_test, columns=features_continuous + features_categorical) \
        .merge(y_test, left_index=True, right_index=True)

    # Labels must be one-hot encoded for loss='categorical_crossentropy'
    y_train_onehot = to_categorical(y_train, N_CLASSES)
    y_test_onehot = to_categorical(y_test, N_CLASSES)

    # Get neural network architecture and save to disk
    model = create_mlp(input_dim=training_dimension, output_dim=N_CLASSES, arch=HIDDEN_UNITS)

    with open(TRAIN_FILENAME, 'w') as f:
        f.write(model.to_yaml())

    # Output logs to tensorflow TensorBoard
    # tensorboard = TensorBoard()

    # only save model weights for best performing model
    checkpoint = ModelCheckpoint(TRAIN_FILENAME,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    # Stop training early if validation accuracy doesn't improve for long enough
    early_stopping = EarlyStopping(monitor='val_acc', patience=50)

    # Shuffle data for good measure before fitting
    x_train, y_train_onehot = shuffle(x_train, y_train_onehot)

    model.fit(x_train, y_train_onehot,
              nb_epoch=EPOCHS,
              batch_size=TRAIN_BATCH_SIZE,
              shuffle=True,
              callbacks=[checkpoint, early_stopping],
              validation_data=(x_test, y_test_onehot))

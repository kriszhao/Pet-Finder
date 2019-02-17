import itertools
import matplotlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
import matplotlib.pyplot as plt

# CONSTANTS
ITERATIONS = 40000
LABEL = 'AdoptionSpeed'
HIDDEN_UNITS = [200, 100, 50, 25, 12]
TRAINING_TEST_SPLIT = 0.33
RANDOM_NUMBER_SEED = 42


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

    return final_data, data_categorical, data_continuous, pet_id


def input_function(data_set, training=True):
    continuous_cols = {key: tf.constant(data_set[key].values) for key in features_continuous}

    categorical_cols = {
        key: tf.SparseTensor(indices=[[i, 0] for i in range(data_set[key].size)],
                             values=data_set[key].values,
                             dense_shape=[data_set[key].size, 1])
        for key in features_categorical}

    # Merges the dictionaries
    feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))

    if training:
        # Convert the label column into a constant Tensor
        label = tf.constant(data_set[LABEL].values)

        return feature_cols, label

    return feature_cols


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    # Import and split
    train, train_categorical, train_continuous, train_pet_id = prepare_data(pd.read_csv('../all/train.csv'))
    test, test_categorical, test_continuous, test_pet_id = prepare_data(pd.read_csv('../all/test/test.csv'))

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

    # Deep neural network model
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                              activation_fn=tf.nn.relu,
                                              hidden_units=HIDDEN_UNITS)

    # Train model
    regressor.fit(input_fn=lambda: input_function(x_train), steps=ITERATIONS)

    # Evaluate model
    evaluation = regressor.evaluate(input_fn=lambda: input_function(x_test, training=True), steps=1)

    # Predictions
    y = regressor.predict(input_fn=lambda: input_function(x_test))
    predictions = list(itertools.islice(y, x_test.shape[0]))
    predictions = pd.DataFrame(y_scaler.inverse_transform(np.array(predictions).reshape(len(predictions), 1)))

    # Compute accuracy
    rounded_predictions = predictions.round()
    reality = pd.DataFrame(train_scaler.inverse_transform(x_test), columns=[columns])[LABEL]
    matching = rounded_predictions.where(reality.values == rounded_predictions.values)
    accuracy = matching.count()[0] / len(reality) * 100

    # Print metrics
    print('Final loss on testing set: {0:f}'.format(evaluation['loss']))
    print('Final accuracy: {0:.2f}%'.format(accuracy))

    # Plot final results
    matplotlib.rc('xtick', labelsize=30)
    matplotlib.rc('ytick', labelsize=30)

    fig, ax = plt.subplots(figsize=(50, 40))

    plt.style.use('ggplot')
    plt.plot(predictions.values, reality.values, 'ro')
    plt.xlabel('Predictions', fontsize=30)
    plt.ylabel('Reality', fontsize=30)
    plt.title('Predictions x Reality on dataset Test', fontsize=30)
    ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)

    plt.show()

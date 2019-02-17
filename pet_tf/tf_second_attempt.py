import itertools
import matplotlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
import matplotlib.pyplot as plt


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

    # Fill missing numerical data
    data_numerical = data.select_dtypes(exclude=['object'])
    data_numerical.fillna(0, inplace=True)

    # Fill missing string data
    data_categorical = data.select_dtypes(include=['object'])
    data_categorical.fillna('NONE', inplace=True)

    final_data = data_numerical.merge(data_categorical, left_index=True, right_index=True)
    return final_data, data_categorical, data_numerical, pet_id


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    ITERATIONS = 50000

    # Import and split
    train, train_categorical, train_numerical, train_pet_id = prepare_data(pd.read_csv('../all/train.csv'))
    test, test_categorical, test_numerical, test_pet_id = prepare_data(pd.read_csv('../all/test/test.csv'))

    # Remove the outliers
    clf = IsolationForest(max_samples=100, random_state=42)
    clf.fit(train_numerical)
    y_no_outliers = clf.predict(train_numerical)
    y_no_outliers = pd.DataFrame(y_no_outliers, columns=['Top'])

    train_numerical = train_numerical.iloc[y_no_outliers[y_no_outliers['Top'] == 1].index.values]
    train_numerical.reset_index(drop=True, inplace=True)

    train_categorical = train_categorical.iloc[y_no_outliers[y_no_outliers['Top'] == 1].index.values]
    train_categorical.reset_index(drop=True, inplace=True)

    train = train.iloc[y_no_outliers[y_no_outliers['Top'] == 1].index.values]
    train.reset_index(drop=True, inplace=True)

    col_train_num = list(train_numerical.columns)
    col_train_num_no_label = list(train_numerical.columns)

    col_train_cat = list(train_categorical.columns)

    col_train_num_no_label.remove('AdoptionSpeed')

    matrix_train = np.matrix(train_numerical)
    matrix_test = np.matrix(test_numerical)
    mat_new = np.matrix(train_numerical.drop('AdoptionSpeed', axis=1))
    matrix_y = np.array(train.AdoptionSpeed)

    prepro_y = MinMaxScaler()
    prepro_y.fit(matrix_y.reshape(matrix_y.shape[0], 1))

    prepro = MinMaxScaler()
    prepro.fit(matrix_train)

    prepro_test = MinMaxScaler()
    prepro_test.fit(mat_new)

    train_num_scale = pd.DataFrame(prepro.transform(matrix_train), columns=col_train_num)
    test_num_scale = pd.DataFrame(prepro_test.transform(matrix_test), columns=col_train_num_no_label)

    train[col_train_num] = pd.DataFrame(prepro.transform(matrix_train), columns=col_train_num)
    test[col_train_num_no_label] = test_num_scale

    # List of features
    COLUMNS = col_train_num
    FEATURES = col_train_num_no_label
    LABEL = "AdoptionSpeed"

    FEATURES_CAT = col_train_cat

    engineered_features = []

    for continuous_feature in FEATURES:
        engineered_features.append(tf.contrib.layers.real_valued_column(continuous_feature))

    for categorical_feature in FEATURES_CAT:
        sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(categorical_feature, hash_bucket_size=1000)

        engineered_features.append( tf.contrib.layers.embedding_column(sparse_id_column=sparse_column,
                                                                       dimension=16,
                                                                       combiner="sum"))

    # Training set and Prediction set with the features to predict
    training_set = train[FEATURES + FEATURES_CAT]
    prediction_set = train.AdoptionSpeed

    # Train and Test
    x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES + FEATURES_CAT],
                                                        prediction_set,
                                                        test_size=0.33,
                                                        random_state=42)

    y_train = pd.DataFrame(y_train, columns=[LABEL])
    training_set = pd.DataFrame(x_train, columns=FEATURES + FEATURES_CAT).merge(y_train, left_index=True,
                                                                                right_index=True)

    # Training for submission
    training_sub = training_set[FEATURES + FEATURES_CAT]
    testing_sub = test[FEATURES + FEATURES_CAT]

    # Same thing but for the test set
    y_test = pd.DataFrame(y_test, columns=[LABEL])
    testing_set = pd.DataFrame(x_test, columns=FEATURES + FEATURES_CAT).merge(y_test, left_index=True, right_index=True)

    training_set[FEATURES_CAT] = training_set[FEATURES_CAT].applymap(str)
    testing_set[FEATURES_CAT] = testing_set[FEATURES_CAT].applymap(str)


    def input_fn_new(data_set, training=True):
        continuous_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}

        categorical_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(data_set[k].size)], values=data_set[k].values,
            dense_shape=[data_set[k].size, 1]) for k in FEATURES_CAT}

        # Merges the two dictionaries into one.
        feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))

        if training:
            # Converts the label column into a constant Tensor.
            label = tf.constant(data_set[LABEL].values)

            # Returns the feature columns and the label.
            return feature_cols, label

        return feature_cols


    # Model
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                              activation_fn=tf.nn.relu,
                                              hidden_units=[200, 100, 50, 25, 12])

    categorical_cols = {
        k: tf.SparseTensor(indices=[[i, 0] for i in range(training_set[k].size)], values=training_set[k].values,
                           dense_shape=[training_set[k].size, 1]) for k in FEATURES_CAT}

    # Deep Neural Network Regressor with the training set which contain the data split by train test split
    regressor.fit(input_fn=lambda: input_fn_new(training_set), steps=ITERATIONS)

    ev = regressor.evaluate(input_fn=lambda: input_fn_new(testing_set, training=True), steps=1)

    loss_score4 = ev["loss"]
    print("Final Loss on the testing set: {0:f}".format(loss_score4))

    # Predictions
    y = regressor.predict(input_fn=lambda: input_fn_new(testing_set))
    predictions = list(itertools.islice(y, testing_set.shape[0]))
    predictions = pd.DataFrame(prepro_y.inverse_transform(np.array(predictions).reshape(len(predictions), 1)))

    reality = pd.DataFrame(prepro.inverse_transform(testing_set), columns=[COLUMNS]).AdoptionSpeed
    rounded_predictions = predictions.round()

    matching = rounded_predictions.where(reality.values == rounded_predictions.values)
    accuracy = matching.count()[0] / len(reality) * 100

    print('Accuracy: {0:.2f}%'.format(accuracy))

    # matplotlib.rc('xtick', labelsize=30)
    # matplotlib.rc('ytick', labelsize=30)
    #
    # fig, ax = plt.subplots(figsize=(50, 40))
    #
    # plt.style.use('ggplot')
    # plt.plot(predictions.values, reality.values, 'ro')
    # plt.xlabel('Predictions', fontsize=30)
    # plt.ylabel('Reality', fontsize=30)
    # plt.title('Predictions x Reality on dataset Test', fontsize=30)
    # ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
    # plt.show()

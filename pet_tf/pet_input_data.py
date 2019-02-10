import numpy
import pandas as pd

from sklearn import preprocessing


def extract_pet_data(path, is_test=False):
    pet_data_df = pd.read_csv(path, sep=',')

    if not is_test:
        pet_data_df = pet_data_df.drop(['RescuerID', 'Description', 'PetID', 'AdoptionSpeed', 'State'], axis=1)

    # Apply binning to ages
    pet_data_df['Age'] = pd.cut(pet_data_df['Age'], [-1, 2, 3, 6, 255], labels=[0, 1, 2, 3])

    # Replace names with 1 is present, 0 if not present
    pet_data_df.loc[pet_data_df['Name'].notnull(), 'Name'] = 1
    pet_data_df.loc[pet_data_df['Name'].isnull(), 'Name'] = 0

    factorized_pet_df = pet_data_df.apply(lambda col: pd.factorize(col, sort=True)[0])
    factorized_values = factorized_pet_df.values
    factorized_scaled = preprocessing.MinMaxScaler().fit_transform(factorized_values)
    normalized_pet_df = pd.DataFrame(factorized_scaled)

    return normalized_pet_df


def extract_pet_labels(path, num_classes):
    labels = pd.read_csv(path, sep=',')['AdoptionSpeed']
    one_hot_labels = one_hot_labels_conversion(labels, num_classes)
    return one_hot_labels


def one_hot_labels_conversion(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


class DataSet(object):
    def __init__(self, pet_data, labels):
        assert pet_data.shape[0] == labels.shape[0], (
            'pet_data.shape: {} labels.shape: {}'.format(pet_data.shape, labels.shape))

        self._num_examples = pet_data.shape[0]

        self._pet_data = pet_data
        self._labels = pd.DataFrame(labels)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def pet_data(self):
        return self._pet_data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start_epoch = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            self._pet_data = self._pet_data.sample(self._num_examples)
            self._labels = self._labels.sample(self._num_examples)

            start_epoch = 0
            self._index_in_epoch = batch_size

            assert batch_size <= self._num_examples

        end_epoch = self._index_in_epoch

        return self._pet_data[start_epoch:end_epoch], self._labels[start_epoch:end_epoch]


def read_data_sets():
    class DataSets(object):
        pass

    # TODO: get this programmatically
    num_labels = 5

    data_sets = DataSets()

    train_pet_data = extract_pet_data('./all/train.csv')
    train_labels = extract_pet_labels('./all/train.csv', num_labels)

    test_pet_data = extract_pet_data('./all/test/test.csv', is_test=True)

    # TODO: figure out what to do with missing test labels
    test_labels = extract_pet_labels('./all/train.csv', num_labels)

    validation_size = round(0.7 * len(train_labels))

    validation_pet_data = train_pet_data[:validation_size]
    validation_labels = train_labels[:validation_size]

    train_pet_data = train_pet_data[validation_size:]
    train_labels = train_labels[validation_size:]

    data_sets.train = DataSet(train_pet_data, train_labels)
    data_sets.validation = DataSet(validation_pet_data, validation_labels)

    # TODO: figure out test data, since there are no labels
    data_sets.test = DataSet(validation_pet_data, validation_labels)

    return data_sets

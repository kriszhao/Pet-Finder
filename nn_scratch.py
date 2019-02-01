# Code taken from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp


# Load a CSV file
def load_csv(filename):
    dataset = list()

    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()

    for i, value in enumerate(unique):
        lookup[value] = i

    for row in dataset:
        row[column] = lookup[row[column]]

    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    return [[min(column), max(column)] for column in zip(*dataset)]


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()

        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))

        dataset_split.append(fold)

    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0

    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()

    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()

        for batch in fold:
            batch_copy = list(batch)
            test_set.append(batch_copy)
            batch_copy[-1] = None

        predicted = algorithm(train_set, test_set, *args)
        actual = [batch[-1] for batch in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]

    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]

    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, batch):
    inputs = batch

    for layer in network:
        # Outputs of the previous layer become inputs of the next layer
        new_inputs = []

        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])

        inputs = new_inputs

    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# back propagate error and store in neurons
def backward_propagate_error(network, expected):
    # Process layers backwards
    for i in reversed(range(len(network))):
        layer = network[i]

        # Calculate error for output layer
        if i == len(network) - 1:
            for h in range(len(layer)):
                neuron = layer[h]

                # t_k - o_k
                error = expected[h] - neuron['output']

                # (t_k - o_k) * (o_k * (1 - o_k))
                neuron['delta'] = error * transfer_derivative(neuron['output'])

        # Calculate error for hidden layer
        else:
            for h in range(len(layer)):
                neuron = layer[h]

                # sum_k(w_h,k * d_k)
                error = sum(neuron['weights'][h] * neuron['delta'] for neuron in network[i + 1])
                neuron['delta'] = error * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, batch, learning_rate):
    for i in range(len(network)):
        inputs = batch[:-1]

        # Get all inputs (which are the outputs of the previous layer)
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]

        for neuron in network[i]:
            # w_i,j = w_i,j + Δw_i,j
            for weight_index in range(len(inputs)):
                neuron['weights'][weight_index] += learning_rate * neuron['delta'] * inputs[weight_index]

            # w_i,j = w_i,j + Δw_i,j
            neuron['weights'][-1] += learning_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, learning_rate, epochs, n_outputs):
    for epoch in range(epochs):
        if epoch % 100 == 0:
            print('Now in epoch {}'.format(epoch))

        for batch in train:
            forward_propagate(network, batch)
            expected = [0 for _ in range(n_outputs)]
            expected[batch[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, batch, learning_rate)


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
    network.append(output_layer)
    return network


# Make a prediction with a network
def predict(network, batch):
    outputs = forward_propagate(network, batch)
    return outputs.index(max(outputs))


# back propagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, learning_rate, epochs, hidden_layers):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([batch[-1] for batch in train]))

    network = initialize_network(n_inputs, hidden_layers, n_outputs)
    print('Network initialized with {0} inputs, {1} hidden layers, and {2} outputs'
          .format(n_inputs, hidden_layers, n_outputs))

    train_network(network, train, learning_rate, epochs, n_outputs)
    predictions = list()

    for batch in test:
        prediction = predict(network, batch)
        predictions.append(prediction)

    return predictions


if __name__ == '__main__':
    # Test back propagation on Seeds dataset
    seed(1)

    # load and prepare data
    filename = 'seeds_dataset.csv'
    dataset = load_csv(filename)

    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)

    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)

    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)

    # evaluate algorithm
    n_folds = 5
    l_rate = 0.1
    n_epoch = 500
    n_hidden = 3

    scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    print('Scores: {}'.format(scores))
    print('Mean Accuracy: {:.3f}%'.format(sum(scores) / float(len(scores))))

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    tf.enable_eager_execution()

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    train_dataset_path = "../all/train.csv"

    # column order in CSV file
    column_names = ['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
                    'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
                    'RescuerID', 'VideoAmt', 'Description', 'PetID', 'PhotoAmt', 'AdoptionSpeed']

    feature_names = column_names[:-1]
    label_name = column_names[-1]

    print("Features: {}".format(feature_names))
    print("Label: {}".format(label_name))

    class_names = ['0', '1', '2', '3', '4']

    batch_size = 32

    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_path,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1)


    def pack_features_vector(features, labels):
        """Pack the features into a single array."""
        features = tf.stack(list(features.values()), axis=1)
        return features, labels


    train_dataset = train_dataset.map(pack_features_vector)

    features, labels = next(iter(train_dataset))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(len(feature_names),)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(len(label_name))
    ])


    def loss(model, x, y):
        y_ = model(x)
        return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


    l = loss(model, features, labels)
    print("Loss test: {}".format(l))


    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    global_step = tf.Variable(0)

    loss_value, grads = grad(model, features, labels)

    print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                              loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

    print("Step: {}, Loss: {}".format(global_step.numpy(), loss(model, features, labels).numpy()))

    from tensorflow import contrib

    tfe = contrib.eager

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)

    fig.show()

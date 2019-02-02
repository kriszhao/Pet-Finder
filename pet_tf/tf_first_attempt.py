import tensorflow as tf

import pet_tf.pet_input_data as input_data

pet_data = input_data.read_data_sets()

# Set parameters
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

# TODO: get programmatically
num_inputs = 19
num_outputs = 5

# TF graph input
x = tf.placeholder('float', [None, num_inputs])
y = tf.placeholder('float', [None, num_outputs])

# Create a model

# Set model weights
W = tf.Variable(tf.zeros([num_inputs, num_outputs]))
b = tf.Variable(tf.zeros([num_outputs]))

# Construct a linear model
with tf.name_scope('Wx_b') as scope:
    model = tf.nn.softmax(tf.matmul(x, W) + b)

# Add summary ops to collect data
w_h = tf.summary.histogram('weights', W)
b_h = tf.summary.histogram('biases', b)

# More name scopes will clean up graph representation
with tf.name_scope('cost_function'):
    # Minimize error using logit function
    cost_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.matmul(x, W) + b, labels=y))

    # Create a summary to monitor the cost function
    tf.summary.scalar('cost_function', cost_function)

with tf.name_scope('train'):
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initializing the variables
init = tf.initialize_all_variables()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Change this to a location on your computer
    summary_writer = tf.summary.FileWriter('data/logs', graph_def=sess.graph_def)

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(pet_data.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = pet_data.train.next_batch(batch_size)

            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

            # Write logs for each iteration
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration * total_batch + i)

        # Display logs per iteration step
        if iteration % display_step == 0:
            print('Iteration:', '%04d' % (iteration + 1), 'cost=', '{:.9f}'.format(avg_cost))

    print('Tuning completed!')

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
    print('Accuracy:', accuracy.eval({x: pet_data.test.pet_data, y: pet_data.test.labels}))

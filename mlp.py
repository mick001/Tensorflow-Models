#!/usr/bin/python3

"""
Created on Thu Sep  8 23:02:19 2016

@author: Michy
@description:
    
    A customizable model for a deep MLP in Tensorflow.
    Works for binary and multilabel classification tasks.
    Works with data in .csv format as follows:
    
    x1,x2,...,y
    0,1,...,0
    
    Before running, you MUST set:
    - FILE_PATH path to a .csv file containing the data.
    - Y_LABEL label of the variable to be predicted    
    
    Then you might want to set the network parameters.

"""

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split

FILE_PATH = '~/data.csv'                                # Path to .csv dataset
raw_data = pd.read_csv(FILE_PATH)						# Open raw .csv

print("Raw data loaded successfully...\n")
#------------------------------------------------------------------------------
# Variables

Y_LABEL = 'y'                                   		    	# Name of the variable to be predicted
KEYS = [i for i in raw_data.keys().tolist() if i != Y_LABEL]	# Name of predictors
N_INSTANCES = raw_data.shape[0]                     			# Number of instances
N_INPUT = raw_data.shape[1] - 1                     			# Input size
N_CLASSES = raw_data[Y_LABEL].unique().shape[0]     			# Number of classes (output size)
TEST_SIZE = 0.1                                    			    # Test set size (% of train set)
TRAIN_SIZE = int(N_INSTANCES * (1 - TEST_SIZE))     			# Train size
LEARNING_RATE = 0.001                               			# Learning rate
TRAINING_EPOCHS = 300                               			# Number of epochs
BATCH_SIZE = 100                                    			# Batch size
DISPLAY_STEP = 1                                    			# Display progress each x epochs
HIDDEN_SIZE = 200	                                   	    	# Number of hidden neurons
ACTIVATION_FUNCTION_INLAYER = tf.nn.tanh                        # In-layer act fct
ACTIVATION_FUNCTION_OUT = tf.nn.tanh                            # Last layer act fct
STDDEV = 0.1                                        			# Standard deviation (for weights random init)
RANDOM_STATE = 100 									            # Random state for train_test_split

print("Variables loaded successfully...\n")
print("Number of predictors \t%s" %(N_INPUT))
print("Number of classes \t%s" %(N_CLASSES))
print("Number of instances \t%s" %(N_INSTANCES))
print("\n")
#------------------------------------------------------------------------------
# Loading data

# Load data
data = raw_data[KEYS].get_values()                  			# X data
labels = raw_data[Y_LABEL].get_values()             			# y data

# One hot encoding for labels
labels_ = np.zeros((N_INSTANCES, N_CLASSES))
labels_[np.arange(N_INSTANCES), labels] = 1

# Train-test split
data_train, data_test, labels_train, labels_test = train_test_split(data,
                                                                    labels_,
                                                                    test_size = TEST_SIZE,
                                                                    random_state = RANDOM_STATE)

print("Data loaded and splitted successfully...\n")
#------------------------------------------------------------------------------
# Neural net construction

# Net params
n_input = N_INPUT                   # input n labels
n_hidden_1 = HIDDEN_SIZE            # 1st layer
n_hidden_2 = HIDDEN_SIZE            # 2nd layer
n_hidden_3 = HIDDEN_SIZE            # 3rd layer
n_hidden_4 = HIDDEN_SIZE            # 4th layer
n_classes = N_CLASSES               # output m classes

# Tf placeholders
X = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
dropout_keep_prob = tf.placeholder(tf.float32)


def mlp(_X, _weights, _biases, dropout_keep_prob):
    layer1 = tf.nn.dropout(ACTIVATION_FUNCTION_INLAYER(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), dropout_keep_prob)
    layer2 = tf.nn.dropout(ACTIVATION_FUNCTION_INLAYER(tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2'])), dropout_keep_prob)
    layer3 = tf.nn.dropout(ACTIVATION_FUNCTION_INLAYER(tf.add(tf.matmul(layer2, _weights['h3']), _biases['b3'])), dropout_keep_prob)
    layer4 = tf.nn.dropout(ACTIVATION_FUNCTION_INLAYER(tf.add(tf.matmul(layer3, _weights['h4']), _biases['b4'])), dropout_keep_prob)
    out = ACTIVATION_FUNCTION_OUT(tf.add(tf.matmul(layer4, _weights['out']), _biases['out']))
    return out

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=STDDEV)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=STDDEV)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],stddev=STDDEV)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],stddev=STDDEV)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes],stddev=STDDEV)),                                   
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Build model
pred = mlp(X, weights, biases, dropout_keep_prob)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

# Accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize
init = tf.initialize_all_variables()

print("Net built successfully...\n")
print("Starting training...\n")
#------------------------------------------------------------------------------
# Training

# Launch session
sess = tf.Session()
sess.run(init)

# Training loop
for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0.
    total_batch = int(data_train.shape[0] / BATCH_SIZE)
    # Loop over all batches
    for i in range(total_batch):
        randidx = np.random.randint(int(TRAIN_SIZE), size = BATCH_SIZE)
        batch_xs = data_train[randidx, :]
        batch_ys = labels_train[randidx, :]
        # Fit training using batch data
        sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 0.9})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob:1.})/total_batch
    # Display logs per epoch step
    if epoch % DISPLAY_STEP == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost))
        train_acc = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob:1.})
        print ("Training accuracy: %.3f" % (train_acc))


print ("End of training.\n")
print("Testing...\n")
#------------------------------------------------------------------------------
# Testing

test_acc = sess.run(accuracy, feed_dict={X: data_test, y: labels_test, dropout_keep_prob:1.})
print ("Test accuracy: %.3f" % (test_acc))

sess.close()
print("Session closed!")

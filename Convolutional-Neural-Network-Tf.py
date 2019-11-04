# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import argparse
import math
from random import shuffle
import cv2
import scipy
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

#Argparase arguements
ap = argparse.ArgumentParser()
ap.add_argument('-lr', '--lrng_rate', required=True,
                help = 'initial learning rate for gradient descent based algorithms')
ap.add_argument('-batch_size', '--batch_size_', required=True,
                help = 'momentum for momentum based algorithms')
ap.add_argument('-init', '--init_', required=True,
                help = 'number of hidden layers')
ap.add_argument('-save_dir', '--save_dir_', required=True,
                help = 'comma seperated list for sizes of each hidden layer')
ap.add_argument('-epochs', '--epochs_', required=True,
                help = 'comma seperated list for sizes of each hidden layer')
ap.add_argument('-dataAugment', '--data_Augment', required=True,
                help = 'comma seperated list for sizes of each hidden layer')
ap.add_argument('-train', '--train_data', required=True,
                help = 'comma seperated list for sizes of each hidden layer')
ap.add_argument('-test', '--test_data', required=True,
                help = 'comma seperated list for sizes of each hidden layer')
ap.add_argument('-val', '--valid_data', required=True,
                help = 'comma seperated list for sizes of each hidden layer')

args = ap.parse_args()

#############FUNCTION DEFINITION#################################################################################################################
# Function definitions
def conv2d(x, W, b, padding, strides=1):
  # Conv2D wrapper, with bias and relu activation
  if padding == 1:
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

  elif padding == 0:
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k = 2):
        return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'VALID')

def one_hot_encoder(Y):
    y = []
    for i in range(20):
        if i == Y:
            y.append(1)
        else:
            y.append(0)
    return(np.reshape(y, (1, 20)))

def make_label_matrix(Y):

    for i, j in enumerate(Y):
        one_hot = one_hot_encoder(j)
        if i == 0:
            mat_Y = one_hot
        else:
            mat_Y = np.vstack((mat_Y, one_hot))
    return mat_Y

# Data Augmentation
def data_augmentation(X_train):
    def rotation(X_train):
        X_rotate = X_train.copy()
        for i in range(0,X_train.shape[0]):
            angle = np.random.uniform(-15,15)
            X_rotate[i] = scipy.ndimage.rotate(X_train[i], angle = angle, reshape = False)
        return X_rotate

    def flip_horizontal(X_train):
        X_flip_h = X_train.copy()
        for i in range(0,X_train.shape[0]):
            X_flip_h[i] = cv2.flip(X_train[i], 0)
        return X_flip_h

    def flip_vertical(X_train):
        X_flip_v = X_train.copy()
        for i in range(0,X_train.shape[0]):
            X_flip_v[i] = cv2.flip(X_train[i], 0)
        return X_flip_v

    def add_gauss_noise(img):
        img1 = img.copy()
        cv2.randn(img1,0,0.5)
        return(img + img1)

    # Flipping Horizontally
    X_flip_h = flip_horizontal(X_train)

    # Flipping Vertically
    X_flip_v = flip_vertical(X_train)

    # Randomly rotate
    X_rotate = rotation(X_train)

    #Adding Gaussian noise
    X_noise = add_gauss_noise(X_train)

    Augment_X_train = np.concatenate((X_train, X_rotate, X_flip_v, X_noise), axis=0)
    Augment_y_train = np.concatenate((Y_train_r, Y_train_r, Y_train_r, Y_train_r), axis = 0)
    return Augment_X_train, Augment_y_train

def conv_net(x, weights, biases):
    #Convolution Layer - 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides = 1, padding = 1)

    conv1 = tf.contrib.layers.batch_norm(conv1)

    #Convolution Layer - 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], strides = 1, padding = 1)

    # Max Pooling 1
    conv2 = maxpool2d(conv2, k=2)

    conv2 = tf.contrib.layers.batch_norm(conv2)

    # Convolution Layer - 3
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], strides = 1, padding = 1)

    conv3 = tf.contrib.layers.batch_norm(conv3)

    # Convolution Layer - 4
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], strides = 1, padding = 1)

    # Max Pooling 2
    conv4 = maxpool2d(conv4, k=2)

    conv4 = tf.contrib.layers.batch_norm(conv4)

    # Convolution Layer - 5
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], strides = 1, padding = 1)

    conv5 = tf.contrib.layers.batch_norm(conv5)

    conv5 = tf.nn.dropout(conv5, rate = 0.4)

    # Convolution Layer - 6
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'], strides = 1, padding = 0)

    # Max Pooling 3
    conv6 = maxpool2d(conv6, k=2)

    conv6 = tf.contrib.layers.batch_norm(conv6)

    conv6 = tf.nn.dropout(conv6, rate = 0.6)

    # Fully connected layer
    # Reshape conv6 output to fit fully connected layer input
    fc1 = tf.reshape(conv6, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

    fc1 = tf.contrib.layers.batch_norm(fc1)

    fc1 = tf.nn.relu(fc1)

    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    out = tf.contrib.layers.batch_norm(out)

    return out
################################################################################################################################################

print('Reading the data')
train = pd.read_csv(args.train_data)
valid = pd.read_csv(args.valid_data)
test = pd.read_csv(args.test_data)
X_train, Y_train = train.iloc[:, 1:12289].values, train.iloc[:, 12289].values
X_valid, Y_valid = valid.iloc[:, 1:12289].values, valid.iloc[:, 12289].values
X_test = test.iloc[:, 1:12289].values
print("Data read")

Y_train = np.reshape(Y_train, (len(Y_train), 1)) # Adding an axis to Y_train

#size_data = np.shape(X_train)[0]

# Normalizing the data
print("Begin scaling")
sl = StandardScaler()
X_train_norm = sl.fit_transform(X_train)
X_valid_norm = sl.fit_transform(X_valid)
X_test_norm = sl.fit_transform(X_test)

X_train_r = np.reshape(X_train_norm, (-1, 64, 64, 3))
X_test_r = np.reshape(X_test_norm, (-1, 64, 64, 3))
X_valid_r = np.reshape(X_valid_norm, (-1, 64, 64, 3))
print("scaling done")

# Obtain the matrix for labels
#Train
Y_train_r = make_label_matrix(Y_train)
#Valid
Y_valid_r = make_label_matrix(Y_valid)

#size_data = np.shape(X_train)[0]

#Data Augmentation
if args.data_Augment == 1:
    X_train, y_train = data_augmentation(X_train_r)

#Iitialising the hyperparameters
training_iters = int(args.epochs_)
learning_rate = float(args.lrng_rate)
batch_size = int(args.batch_size_)

#Defining the Placeholders
n_input = 64
n_classes = 20
x = tf.placeholder("float", [None, 64, 64, 3])
y = tf.placeholder("float", [None, n_classes])
# Define weights and biases
#Weights & biases dictionary
weights = {
            'wc1': tf.get_variable('W0', shape=(5,5,3,32),
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'wc2': tf.get_variable('W1', shape=(5,5,32,32),
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'wc3': tf.get_variable('W2', shape=(3,3,32,64),
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'wc4': tf.get_variable('W3', shape=(3,3,64,64),
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'wc5': tf.get_variable('W4', shape=(3,3,64,64),
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'wc6': tf.get_variable('W5', shape=(3,3,64,128),
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'wd1': tf.get_variable('W6', shape=(6272,256),
                                   initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'out': tf.get_variable('W7', shape=(256,n_classes),
                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_))
                }

biases = {
            'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'bc2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'bc4': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'bc5': tf.get_variable('B4', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'bc6': tf.get_variable('B5', shape=(128), initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'bd1': tf.get_variable('B6', shape=(256), initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_)),
            'out': tf.get_variable('B7', shape=(n_classes), initializer=tf.contrib.layers.variance_scaling_initializer(factor=args.init_))
                }


#Defining loss functiton and optimizer
pred = conv_net(x , weights, biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

L2_regularize = 0.00001*tf.nn.l2_loss(pred)

cost = tf.reduce_mean(L2_regularize + loss)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#Evaluate Model Node
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Initialize the variables
initialize = tf.global_variables_initializer()

# Shufling the data
list_index = [i for i in range(len(X_train))]
shuffle(list_index)
X_train  = X_train[list_index , : , : , :]
y_train = y_train[list_index , ]


saver = tf.train.Saver()
#Training and Testing the Model
with tf.Session() as sess:
    sess.run(initialize)
    train_loss = []
    validation_loss_list = []
    train_accuracy = []
    validation_accuracy = []
    weights_running = []
    biases_running = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(X_train)//batch_size):
            batch_x = X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
            batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]
      # Run optimization op (backprop).
      # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
            "{:.6f}".format(loss) + ", Training Accuracy= " + \
            "{:.5f}".format(acc))

        print("Optimization Finished!")

    # Calculate accuracy
        validation_acc,validation_loss = sess.run([accuracy,cost], feed_dict={x: X_valid_r,y : Y_valid_r})
        train_loss.append(loss)
        validation_loss_list.append(validation_loss)
        train_accuracy.append(acc)
        validation_accuracy.append(validation_acc)
        print("Validation Accuracy:","{:.5f}".format(validation_acc))
        weights_running.append(weights)
        biases_running.append(biases)

    # Implementation of Early Stopping
        if i > 4:
            if (validation_loss_list[i] >= validation_loss_list[i - 1]) and (validation_loss_list[i - 1] >= validation_loss_list[i - 2]) and (validation_loss_list[i - 2] >= validation_loss_list[i - 3]) and (validation_loss_list[i - 3] >= validation_loss_list[i - 4]) and (validation_loss_list[i - 4] >= validation_loss_list[i - 5]):
                weights = weights_running[i - 5]
                biases = biases_running[i - 5]
    summary_writer.close()
    save_path = saver.save(sess, args.save_dir_)
  # g = tf.get_default_graph()
  # xx = tf.constant(X_train[0])
  # with g.gradient_override_map({'Relu': 'GuidedRelu'}):
  #           #y = tf.nn.relu(x)
  #           z = conv_net[1][6, 6, 10, 10]
  #
  # tf.initialize_all_variables().run()
  #
  # print(tf.gradients(z, x)[0]) #x.eval(), y.eval(), z.eval(), tf.gradients(z, x)[0].eval()
  #
  #Performing the prediction son the test data
    pred_test = sess.run(pred, feed_dict = {x : X_test_r})

    final_pred = np.argmax(pred_test, axis = 1)

    final_pred = pd.DataFrame(final_pred)

    #Creating the excel shhet as per the required format
    id_ = []
    for u in range(len(final_pred)):
        id_.append(u)

    id_ = pd.DataFrame(id_)

    final = pd.concat([id_, final_pred], axis = 1)
    final.columns = ['id', 'label']
    #final.to_csv("G:\\ADKIITM\\Sem8\\Deep Learning\\Programming Assignment 2\\final_predictions.csv", index = False)

    final.to_csv(args.save_dir_, index = False)

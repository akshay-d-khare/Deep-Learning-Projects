# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import argparse
from random import shuffle

%matplotlib inline

ap = argparse.ArgumentParser()
ap.add_argument('-train', '--train_data', required=True,
                help = 'path to the training data')
ap.add_argument('-test', '--test_data', required=True,
                help = 'path to the test data')

args = ap.parse_args()


def sigmoid(x):
  return(1/(1+np.exp(-x)))

# Load the data
training_data = pd.read_csv(args.train_data)
test_data = pd.read_csv(args.test_data)

# Set the seed
random.seed(10)

# Splitting the data
X_train = training_data.drop(['id','label'], axis = 1)
y_train = training_data['label']
X_test = test_data.drop(['id','label'], axis = 1)

def shuffle_data(training, X, y):
  train = np.array(training)
  X_array = np.array(X)
  y_array = np.array(y)
  ind_list = [i for i in range(train.shape[0])]
  shuffle(ind_list)
  train  = train[ind_list, :]
  X_array = X_array[ind_list,:]
  y_array = y_array[ind_list]
  return train, X_array, y_array

train, X_train, y_train = shuffle_data(training_data, X_train, y_train)

  # Preprocess the data
def convert_binary(X):
  X_final = X.copy()
  X_final[X_final < 127] = 0
  X_final[X_final>=127] = 1
  return X_final

X_train_final = convert_binary(X_train)
X_test_final = convert_binary(X_test)

# Store as numpy arrays
trainX = np.array(X_train_final)
testX = np.array(X_test_final)

# Number of input layer neurons
j = X_train_final.shape[1]

# Number of training examples
m = X_train_final.shape[0]

# Initializing the hyperparameters
hidden_units = 200
lr = 0.001
k = 20
batch_size = 1
n_epoch = 5

# Parameter initialisation
weights = np.reshape(np.random.normal(0,0.01, size = (j*hidden_units)), newshape = (j,hidden_units))
c = np.reshape(np.random.normal(0,0.01, size = (batch_size*hidden_units)), newshape = (batch_size,hidden_units))
b = np.reshape(np.random.normal(0,0.01, size = (batch_size*j)), newshape = (batch_size,j))

# Create representations of visible and hidden nodes
v_full = np.zeros(m*j).reshape(m,j)
h_full = np.zeros(m*hidden_units).reshape(m,hidden_units)

n_batch = int(trainX.shape[0]/batch_size)

# Initialize reconstruction error
error_list = []
error_step = 0

q3_samples = []

# Training the RBM
err_iter = []
for epoch in range(n_epoch):
  error_step = 0
  for batch_i in range(n_batch):
    v = trainX[(batch_i*batch_size):((batch_i+1)*batch_size),:]
    v_hat = v
    for t in range(k):
      # Sample hidden representations for each batch
      h = sigmoid(np.matmul(v_hat,weights)+c)

      # Convert the hidden node values to stohastic binary states
      threshold = np.random.uniform(0,1)
      h[h > threshold] = 1
      h[h < threshold] = 0

      # Reconstruct the visible node values
      v_hat = sigmoid(np.matmul(h,weights.T)+b)

    # Updating parameters
    weights = weights + lr*((np.matmul(v.T,sigmoid(np.matmul(v,weights)+c))) - (np.matmul(v_hat.T,sigmoid(np.matmul(v_hat,weights)+c))))
    b = b + lr*(v-v_hat)
    c = c + lr*((sigmoid(np.matmul(v,weights)+c)) - (sigmoid(np.matmul(v_hat,weights)+c)))


    #Store the hidden representations and the reconstructed input
    v_full[(batch_i*batch_size):((batch_i+1)*batch_size),:] = v_hat
    h_full[(batch_i*batch_size):((batch_i+1)*batch_size),:] = h


  #Calculating reconstruction error every 5000 step
    if batch_i%5000 == 0:
      error_step = (np.square(v-v_hat).sum())/batch_size
      err_iter.append(error_step)
    # Every 100 steps store the samples generated in a list for Q3 assuming SGD converges in 6400 iterations
    if batch_i<=6400:
      if (batch_i+1)%100 == 0:
        q3_samples.append(v_hat)

  #Store error over an epoch
  error_list.append(error_step)
  print("Epoch %d finished" % epoch)

#Code for Q1
#computing the hidden representations

def hidden_represent(weights, c):
#    hidd = np.zeros((len(X_test), 100))

    for co, v in enumerate(testX):
        v = np.reshape(v, (784, 1))
        h = sigmoid(np.matmul(weights, v) + c)
        if co == 0:
            hidd = np.vstack([h, h])
        else:
            hidd = np.vstack([hidd, h])

    return hidd

hid_rep = hidden_represent(weights, c)

hid_rep = hid_rep[1:]

from sklearn.manifold import TSNE
import time
time_start = time.time()
RS = 123
hid_rep1 = hid_rep[0 : 10000, :]
fashion_tsne = TSNE(random_state=RS).fit_transform(hid_rep1)


print ("t-SNE done! Time elapsed: {} seconds".format(time.time()-time_start))

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
import seaborn as sns
import matplotlib.patheffects as PathEffects

y_subset = Y_test[0 : 10000]
fashion_scatter(fashion_tsne, y_subset)

#Code to obtain Q2 plots
loss_1 = []
loss_3 = []
loss_5 = []
loss_15 = []
loss_20 = []
for i in [12, 23, 35, 47, 59]:
  loss_1.append(err1[i])
for i in [12, 23, 35, 47, 59]:
  loss_3.append(err_3[i])
for i in [12, 23, 35, 47, 59]:
  loss_5.append(err_5[i])
for i in [12, 23, 35, 47, 59]:
  loss_15.append(err_15[i])
for i in [12, 23, 35, 47, 59]:
  loss_20.append(err_20[i])

plt.plot(range(1, 6), loss_1, color = 'red', label = 'K = 1')

plt.plot(range(1, 6), loss_3, color = 'blue', label = 'K = 3')

plt.plot(range(1, 6), loss_5, color = 'green', label = 'K = 5')

plt.plot(range(1, 6), loss_15, color = 'black', label = 'K = 15')

plt.plot(range(1, 6), loss_20, color = 'brown', label = 'K = 20')

plt.xlabel('Number of iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Variation of the loss with epochs for different values of k')

#Code to obtain Q3 plots
Question 3
fig_q3 = plt.figure(figsize=(15,15))
fig_q3.subplots_adjust(hspace=1, wspace=0.5)
for i in range(1, 65):
    ax = fig_q3.add_subplot(8, 8, i)
    ax.imshow(q3_samples[i-1].reshape(28,28),cmap = 'gray')

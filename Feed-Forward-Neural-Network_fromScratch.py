# Programming Assignment 1 (Backpropogation)

import pandas as pd
import argparse
import numpy as np
import pickle
import numpy as np
import os.path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ap = argparse.ArgumentParser()
ap.add_argument('-lr', '--lrng_rate', required=True,
                help = 'initial learning rate for gradient descent based algorithms')
ap.add_argument('-momentum', '--moment', required=True,
                help = 'momentum for momentum based algorithms')
ap.add_argument('-num_hidden', '--n_hidden_layers', required=True,
                help = 'number of hidden layers')
ap.add_argument('-sizes', '--size_hidden', required=True,
                help = 'comma seperated list for sizes of each hidden layer')
ap.add_argument('-activation', '--act_function', required=True,
                help = 'choice of the activation function (tanh/sigmoid)')
ap.add_argument('-loss', '--loss_function', required=True,
                help = 'loss function squared error/cross entropy' )
ap.add_argument('-opt', '--opt_algorithm', required=True,
                help = 'The optimization algorithm to be used')
ap.add_argument('-batch_size', '--n_batch_size', required=True,
                help = 'batch size to be used - valid values are 1 and multiples of 5')
ap.add_argument('-epochs', '--n_epochs', required=True,
                help = 'the number of passes over the data')
ap.add_argument('-anneal', '--anneal_', required=True,
                help = 'Takes the value true if user wishes to anneal the learning rate')
ap.add_argument('-save_dir', '--save_directory', required=True,
                help = 'directory where pickled model should be saved')
ap.add_argument('-expt_dir', '--expt_directory', required=True,
                help = 'directory where log files should be saved')
ap.add_argument('-train', '--train_dataset', required=True, help = 'path to the training dataset')
ap.add_argument('-val', '--validation_dataset', required=True,
                help = 'path to the validation dataset')
ap.add_argument('-test', '--test_dataset', required=True,
                help = 'path to the test dataset')
ap.add_argument('-pretrain', '--pre_train', required=True,
                help = 'kj')
ap.add_argument('-state', '--n_state', required=True,
                help = 'nk')
ap.add_argument('-testing', '--testing_', required=True,
                help = 'm')

args = ap.parse_args()

#---------------------------------------------------------------------------------------------------------------------------------------


def save_weights(list_of_weights, epoch):
    with open(args.save_directory +'\weights_{}.pkl'.format(epoch), 'wb') as f:
             pickle.dump(list_of_weights, f)

def load_weights(state):
    with open(args.save_directory +'\weights_{}.pkl'.format(state), 'rb') as f:
        list_of_weights = pickle.load(f)
    return list_of_weights

#Function definitions
def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def tanh(x):
    return ((np.exp(2*x) - 1)/(np.exp(2*x) + 1))

def init_weights_biases(n_hidden_layers, size_hidden, init):
    Weights = dict()
    Biases = dict()

    layer_dims = []
    layer_dims.append(100)

    size_hidden1 = []
    size_hidden2 = []
    size_hidden1 = size_hidden.split(',')

    for k in size_hidden1:
        size_hidden2.append(int(k))

    for j in size_hidden2:
        layer_dims.append(j)
    layer_dims.append(10)

    for i in range(1,  n_hidden_layers + 2):
        np.random.seed(1234)
        Weights[i] = np.random.randn( layer_dims[i], layer_dims[i-1])*np.sqrt(2/(layer_dims[i] + layer_dims[i-1]))
        Biases[i] = np.random.randn(layer_dims[i], 1)*np.sqrt(2/(layer_dims[i] + layer_dims[i-1]))
    return Weights, Biases


# Forward Propagation
def forward_propogation(X, Y, Weights, Biases, act_function, loss_function, n_hidden_layers):
    a = dict()
    h = dict()
    X = np.reshape(X, (100, 1))
    a_old = X

# Comstructing a one hot vector for Y
    Y_one_hot = []
    for j in range(10):
        if j == Y:
            Y_one_hot.append(1)
        else:
            Y_one_hot.append(0)
    Y_one_hot = np.resize(Y_one_hot, (10, 1))

# Computing the pre-activation and activation for hidden layers
    for j in range(1, n_hidden_layers + 1):
        a_new = Biases[j] + np.matmul(Weights[j], a_old)
        a[j] = a_new
        if act_function  == "sigmoid":
            h_ = sigmoid(a_new)
        else:
            h_ = tanh(a_new)
        a_old = h_
        h[j] = h_

#Separate compution of the preactivation and activation for the outut layer
    a[n_hidden_layers + 1] = Biases[n_hidden_layers + 1] + np.matmul(Weights[n_hidden_layers + 1], h[n_hidden_layers])

    y_pred = np.exp(a[n_hidden_layers + 1] - np.max(a[n_hidden_layers + 1]))/(np.sum(np.exp(a[n_hidden_layers + 1] - np.max(a[n_hidden_layers + 1]))))

# Computing the loss function
    if loss_function == "sq":
        loss = np.sum(0.5 * ( Y_one_hot - y_pred )**2)
    else:
        loss = -np.sum(Y_one_hot*np.log(y_pred))

    return loss, a, h, y_pred

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def tanh(x):
    return ((np.exp(2*x) - 1)/(np.exp(2*x) + 1))

# Back Propagation
def back_propagation(loss_function, act_function, loss, a, h, y_pred, Weights, X, Y, n_hidden_layers): # NOTE: X should be a row and Y a number
    X = np.reshape(X, (100, 1))
    one_hot = []
    if loss_function == "ce":
        for j in range(10):
            if j == Y:
                one_hot.append(1)
            else:
                one_hot.append(0)
        one_hot = np.resize(one_hot, (10, 1))

# Gradient of Loss wrt final predicted probabilities
        grad_L_y_pred = one_hot/ (-y_pred[Y])

# Gradient of Loss wrt final output pre-activation
        grad_L_a_out = -(one_hot - y_pred)

    else:
        for j in range(10):
            if j == Y:
                one_hot.append(1)
            else:
                one_hot.append(0)

# Gradient of Loss wrt final predicted probabilities
        one_hot = np.resize(one_hot, (10, 1))

        grad_L_y_pred = -2*(one_hot - y_pred)

# Gradient of Loss wrt final output pre-activation
        grad_L_a_out = np.matmul((np.eye(10) - np.matmul(y_pred , np.matrix.transpose(y_pred))), grad_L_y_pred)

# Initialse the dictionaries to store the gradients of the loss function wrt Weights and Biases
    grad_Weights = dict()
    grad_Biases = dict()

# Assign the gradients of loss function wrt weights and biases of the output layer
    grad_Biases[n_hidden_layers + 1] = grad_L_a_out # Since gradient wrt biases for particular layer equals gradient wrt preactivation for that layer
    grad_Weights[n_hidden_layers + 1] = np.matmul(grad_L_a_out, np.matrix.transpose(h[n_hidden_layers]))

# Initialize the grad wrt upper layer pre-activation to compute the grad wrt the activation of the layer below
    grad_L_a_ul = grad_L_a_out

# Create a couter which is representative of the hidden layer number for the loop below
    flag = n_hidden_layers

# Computing gradient wrt Weights and Biases for all but the first the hidden layers
    while flag > 1:
        grad_L_h = np.matmul( np.matrix.transpose(Weights[flag + 1]), grad_L_a_ul)

        if act_function == "sigmoid":
            grad_L_a = grad_L_h * sigmoid(a[flag]) * (1 - sigmoid(a[flag]))
        else:
            grad_L_a = grad_L_h * (1 - tanh(a[flag]))**2

# Gradient of Loss function wrt Weights of the flag layer
        grad_Weights[flag] = np.matmul(grad_L_a, np.matrix.transpose(h[flag - 1]))

# Gradient of Loss function wrt Wwights of the flag layer
        grad_Biases[flag] = grad_L_a

# Updating the gradient wrt upper preactivation layer for the next iteration
        grad_L_a_ul = grad_L_a

        flag-=1

# Computing gradient wrt Weights and Biases for the first hidden layer
    grad_L_h = np.matmul( np.matrix.transpose(Weights[flag + 1]), grad_L_a_ul)

    if act_function == "sigmoid":
        grad_L_a = grad_L_h * sigmoid(a[flag]) * (1 - sigmoid(a[flag]))
    else:
        grad_L_a = grad_L_h * (1 - tanh(a[flag]))**2
    grad_Weights[flag] = np.matmul(grad_L_a, np.matrix.transpose(X))
    grad_Biases[flag] = grad_L_a

    return grad_Weights, grad_Biases

# Used to make the gradients wrt weights and biases zero
def make_gradients_zero(n_hidden_layers, size_hidden):

    grad_Weights_final = dict()
    grad_Biases_final = dict()
    layer_dims = []
    layer_dims.append(100)

    size_hidden1 = []
    size_hidden2 = []
    size_hidden1 = size_hidden.split(',')


    for k in size_hidden1:
        size_hidden2.append(int(k))

    for j in size_hidden2:
        layer_dims.append(j)
    layer_dims.append(10)


# Initialize the weights and biases to zero
    for i in range(1,  n_hidden_layers + 2):
        grad_Weights_final[i] = np.zeros((layer_dims[i], layer_dims[i-1]))
        grad_Biases_final[i] = np.zeros((layer_dims[i], 1))

    return grad_Weights_final, grad_Biases_final

# Performs Vanilla Gradient Descent
def vanilla_gradient_descent(X_train, Y_train, X_valid, Y_valid, n_hidden_layers, size_hidden, number_epochs, mini_batch_size, lr, act_function, loss_function, anneal,  init):
# Initialize the weights and biases
    if args.pre_train == "True" and args.testing_ == "False":

        learned_params = load_weights(args.n_state)
        w_0 = dict()
        b_0 = dict()

        for tt in range(0, n_hidden_layers + 1):
            w_0[tt] = learned_params[tt]

        for tt in range(n_hidden_layers + 1, len(learned_params) ):
            b_0[tt] = learned_params[tt]


    else:
        w_0, b_0 = init_weights_biases(n_hidden_layers, size_hidden, init)

    error_val = [] # Keeps a track of the validation error at every epoch
    epochs_completed = 0
    Running_Weights = [] # Keeps a track of the weights at each epoch
    Running_Weights.append(w_0)
    Running_Biases = []  # Keeps a track of the biases at each epoch
    Running_Biases.append(b_0)
    loss_training_perstep = 0
    loss_validation_perstep = 0

    # Initiating the log train and log validation files
    completeName_tr = os.path.join(args.expt_directory, "log_train.txt")
    completeName_vl = os.path.join(args.expt_directory, "log_val.txt")
    file_tr = open(completeName_tr, "w")
    file_val = open(completeName_vl, "w")
    n_steps = 0

    while epochs_completed < number_epochs:

        loss_validation = 0 # Validation loss per epoch

        # Set the gradients to zero
        grad_Weights_final, grad_Biases_final = make_gradients_zero(n_hidden_layers, size_hidden)

        n_dpts = 0 # Keeps track of the number of data points considered

        for X, Y in zip(X_train, Y_train):

            loss, a, h, y_pred = forward_propogation(X, Y, Running_Weights[epochs_completed], Running_Biases[epochs_completed], act_function, loss_function, n_hidden_layers)
            # loss_training_perstep += loss
            grad_Weights, grad_Biases = back_propagation(loss_function, act_function, loss, a, h, y_pred, Running_Weights[epochs_completed], X, Y, n_hidden_layers)


            # Update the  gradients of weights and biases
            for k in range(1, n_hidden_layers + 2):
                grad_Weights_final[k] += grad_Weights[k]
                grad_Biases_final[k] += grad_Biases[k]
            n_dpts += 1

            if n_dpts % mini_batch_size == 0:

                n_steps += 1
                for i in range(1, n_hidden_layers + 2):
                    w_0[i] = w_0[i] - lr*(  grad_Weights_final[i] )
                    b_0[i] = b_0[i] - lr*(  grad_Biases_final[i] )


                # Set the gradients to zero
                grad_Weights_final, grad_Biases_final = make_gradients_zero(n_hidden_layers, size_hidden)

                Running_Weights[epochs_completed] = w_0
                Running_Biases[epochs_completed] = b_0

            # Creating the log files for the train and test data
                if n_steps % 100 == 0 and n_steps != 0:

                    error_train = 0 # Training error (accuracy)
                    error_validation = 0 # Validation error (accuracy)
                    y_train = []
                    y_val = []

                # Getting train and validation accuracy
                    for k in X_train:
                        y_train.append(get_final_predictions(k, w_0, b_0, act_function, n_hidden_layers))
                    error_train = np.sum(y_train == Y_train)/len(Y_valid)

                    for k in X_valid:
                        y_val.append(get_final_predictions(k, w_0, b_0, act_function, n_hidden_layers))
                    error_validation = np.sum(y_val == Y_valid)/len(Y_valid)

                # Getting the train and validation losses

                    for X1, Y1 in zip(X_valid, Y_valid):

                        loss1, a1, h1, y_pred1 = forward_propogation(X1, Y1, w_0, b_0, act_function, loss_function, n_hidden_layers)
                        loss_validation_perstep += loss1

                    for X2, Y2 in zip(X_train, Y_train):

                        loss2, a2, h2, y_pred2 = forward_propogation(X2, Y2, w_0, b_0, act_function, loss_function, n_hidden_layers)
                        loss_training_perstep += loss2

                    file_tr.write("Epoch : %f \t" %epochs_completed + "Step : %f \t" %n_steps + "Loss : %.2f \t" %loss_training_perstep + "Error : %.2f \t" %error_train + "lr : %f \n " %lr)

                    file_val.write("Epoch : %f \t" %epochs_completed + "Step : %f \t" %n_steps + "Loss : %.2f \t" %loss_validation_perstep + "Error : %.2f \t" %error_validation + "lr : %f \n " %lr)

                    loss_training_perstep = 0
                    loss_validation_perstep = 0


    # Computing the validation loss after each epoch
        for X3, Y3 in zip(X_valid, Y_valid):

            loss3, a3, h3, y_pred3 = forward_propogation(X3, Y3, w_0, b_0, act_function, loss_function, n_hidden_layers)
            loss_validation += loss3

        error_val.append(loss_validation)
        epochs_completed += 1
        Running_Weights.append(w_0)
        Running_Biases.append(b_0)

        # Saving the weights and biases in pickled form after each epoch
        list_of_parameters = []
        for t in range(1, n_hidden_layers + 2):
            list_of_parameters.append(w_0[t])

        for t in range(1, n_hidden_layers + 2):
            list_of_parameters.append(b_0[t])

        save_weights(list_of_parameters, epochs_completed)

        if anneal == "True":
        # Annealing of the learning rate
            if epochs_completed > 1:
                if error_val[epochs_completed - 1] > error_val[epochs_completed - 2]:
                    error_val = error_val[:-1] # To rewrite the validation loss entry for the repeated epoch
                    epochs_completed -=1
                    lr = lr*0.5

                    # Assign the weights and biases to the values in the previous epoch
                    w_0 = Running_Weights[epochs_completed]
                    b_0 = Running_Biases[epochs_completed]

    file_tr.close()
    file_val.close()
    return w_0, b_0




# Does Momentum Gradient Descent
def momentum_gradient_descent(X_train, Y_train, X_valid, Y_valid, n_hidden_layers, size_hidden, number_epochs, mini_batch_size, lr, momentum, act_function, loss_function, anneal,  init):
# Initialize the weights and biases
    if args.pre_train == "True" and args.testing_ == "False":

        learned_params = load_weights(args.n_state)
        w_0 = dict()
        b_0 = dict()

        for tt in range(0, n_hidden_layers + 1):
            w_0[tt] = learned_params[tt]

        for tt in range(n_hidden_layers + 1, len(learned_params) ):
            b_0[tt] = learned_params[tt]


    else:
        w_0, b_0 = init_weights_biases(n_hidden_layers, size_hidden, init)

    error_val = [] # Keeps a track of the error at every epoch
    epochs_completed = 0
    Running_Weights = [] # Keeps a track of the weights at each epoch
    Running_Weights.append(w_0)
    Running_Biases = []  # Keeps a track of the biases at each epoch
    Running_Biases.append(b_0)
    update_old_weights, update_old_biases = make_gradients_zero(n_hidden_layers, size_hidden)
    update_new_weights, update_new_biases = make_gradients_zero(n_hidden_layers, size_hidden)
    Running_oldupdate_weights = []
    Running_oldupdate_weights.append(update_old_weights)
    Running_oldupdate_biases = []
    Running_oldupdate_biases.append(update_old_biases)
    loss_validation_perstep = 0 # Validation loss per step
    loss_training_perstep = 0 # Training loss per step
    completeName_tr = os.path.join(args.expt_directory, "log_train.txt")
    completeName_vl = os.path.join(args.expt_directory, "log_val.txt")
    file_tr = open(completeName_tr, "w")
    file_val = open(completeName_vl, "w")
    n_steps = 0

    while epochs_completed < number_epochs:

        # Set the gradients to zero
        grad_Weights_final, grad_Biases_final = make_gradients_zero(n_hidden_layers, size_hidden)
        n_dpts = 0
        loss_validation = 0
        for X, Y in zip(X_train, Y_train):
            n_dpts = n_dpts + 1
            loss, a, h, y_pred = forward_propogation(X, Y, Running_Weights[epochs_completed], Running_Biases[epochs_completed], act_function, loss_function, n_hidden_layers)
            #loss_training_perstep += loss
            grad_Weights, grad_Biases = back_propagation(loss_function, act_function, loss, a, h, y_pred, Running_Weights[epochs_completed], X, Y, n_hidden_layers)


            # Update the  gradients of weights and biases
            for k in range(1, n_hidden_layers + 2):
                grad_Weights_final[k] += grad_Weights[k]
                grad_Biases_final[k] += grad_Biases[k]

            if n_dpts % mini_batch_size == 0:

                n_steps = n_steps + 1
                for i in range(1, n_hidden_layers + 2):
                    update_new_weights[i] = momentum * update_old_weights[i] + lr * grad_Weights_final[i]
                    update_new_biases[i] = momentum * update_old_biases[i] + lr * grad_Biases_final[i]

                    # Assignment of new updates to old updates
                    update_old_weights[i] = update_new_weights[i]
                    update_old_biases[i] = update_new_biases[i]

                for i in range(1, n_hidden_layers + 2):
                    w_0[i] = w_0[i] - update_new_weights[i]
                    b_0[i] = b_0[i] - update_new_biases[i]

                    # Set the gradients to zero
                grad_Weights_final, grad_Biases_final = make_gradients_zero(n_hidden_layers, size_hidden)

                Running_Weights[epochs_completed] = w_0
                Running_Biases[epochs_completed] = b_0


            # Creating the log files for the train and test data
                if n_steps % 100 == 0 and n_steps != 0:

                    error_train = 0 # Training error (accuracy)
                    error_validation = 0 # Validation error (accuracy)
                    y_train = []
                    y_val = []

                # Getting train and validation accuracy
                    for k in X_train:
                        y_train.append(get_final_predictions(k, w_0, b_0, act_function, n_hidden_layers))
                    error_train = np.sum(y_train == Y_train)/len(Y_valid)

                    for k in X_valid:
                        y_val.append(get_final_predictions(k, w_0, b_0, act_function, n_hidden_layers))
                    error_validation = np.sum(y_val == Y_valid)/len(Y_valid)

                # Getting the train and validation losses

                    for X1, Y1 in zip(X_valid, Y_valid):

                        loss1, a1, h1, y_pred1 = forward_propogation(X1, Y1, w_0, b_0, act_function, loss_function, n_hidden_layers)
                        loss_validation_perstep += loss1
                    for X2, Y2 in zip(X_train, Y_train):

                        loss2, a2, h2, y_pred2 = forward_propogation(X2, Y2, w_0, b_0, act_function, loss_function, n_hidden_layers)
                        loss_training_perstep += loss2

                    file_tr.write("Epoch : %f \t" % epochs_completed + "Step : %f \t" % n_steps + "Loss : %.2f \t" %loss_training_perstep + "Error : %.2f \t" %error_train + "lr : %f \n " %lr)

                    file_val.write("Epoch : %f \t" % epochs_completed + "Step : %f \t" % n_steps + "Loss : %.2f \t" %loss_validation_perstep + "Error : %.2f \t" %error_validation + "lr : %f \n " %lr)

                    loss_training_perstep = 0
                    loss_validation_perstep = 0

        # Computing the validation loss after each epoch
        for X3, Y3 in zip(X_valid, Y_valid):

            loss3, a3, h3, y_pred3 = forward_propogation(X3, Y3, w_0, b_0, act_function, loss_function, n_hidden_layers)
            loss_validation += loss3

        error_val.append(loss_validation)
        epochs_completed += 1
        Running_Weights.append(w_0)
        Running_Biases.append(b_0)
        Running_oldupdate_weights.append(update_old_weights)
        Running_oldupdate_biases.append(update_old_biases)

        # Saving the weights and biases in pickled form after each epoch
        list_of_parameters = []
        for t in range(1, n_hidden_layers + 2):
            list_of_parameters.append(w_0[t])

        for t in range(1, n_hidden_layers + 2):
            list_of_parameters.append(b_0[t])

        save_weights(list_of_parameters, epochs_completed)

        # Annealing of the learning rate
        if anneal == "True":
            if epochs_completed > 1:
                if error_val[epochs_completed - 1] > error[epochs_completed - 2]:
                    error_val = error_val[:-1] # To rewrite the validation loss entry for the repeated epoch
                    epochs_completed -=1
                    lr = lr*0.5
# Assign the weights and biases to the values in the previous epoch
                    w_0 = Running_Weights[epochs_completed]
                    b_0 = Running_Biases[epochs_completed]
                    update_old_weights = Running_oldupdate_weights[epochs_completed]
                    update_old_biases = Running_oldupdate_biases[epochs_completed]

    file_tr.close()
    file_val.close()
    return w_0, b_0


# Does Nestorov Accelarated Gradient Descent
def nestorov_accelerated_gradient_descent(X_train, Y_train, X_vaild, Y_valid, n_hidden_layers, size_hidden, number_epochs, mini_batch_size, lr, momentum, act_function, loss_function, anneal, init):
# Initialize the weights and biases
    if args.pre_train == "True" and args.testing_ == "False":

        learned_params = load_weights(args.n_state)
        w_0 = dict()
        b_0 = dict()

        for tt in range(0, n_hidden_layers + 1):
            w_0[tt] = learned_params[tt]

        for tt in range(n_hidden_layers + 1, len(learned_params) ):
            b_0[tt] = learned_params[tt]


    else:
        w_0, b_0 = init_weights_biases(n_hidden_layers, size_hidden, init)

    error_val = [] # Keeps a track of the error at every epoch
    epochs_completed = 0
    Running_Weights = [] # Keeps a track of the weights at each epoch
    Running_Weights.append(w_0)
    Running_Biases = []  # Keeps a track of the biases at each epoch
    Running_Biases.append(b_0)
    update_old_weights, update_old_biases = make_gradients_zero(n_hidden_layers, size_hidden) # Initializing the old updates to zero
    update_new_weights, update_new_biases = make_gradients_zero(n_hidden_layers, size_hidden) # Initializing the new updates to zero
    grad_w_lookahead, grad_b_lookahead = make_gradients_zero(n_hidden_layers, size_hidden) # Initializing the lookahead gradients to zero
    Running_oldupdate_weights = []
    Running_oldupdate_biases = []
    Running_lookahead_weights = []
    Running_lookahead_biases = []
    w_look_ahead = w_0
    b_look_ahead = b_0
    loss_training_perstep = 0
    loss_validation_perstep = 0

    # Initiating the log train and log validation files
    completeName_tr = os.path.join(args.expt_directory, "log_train.txt")
    completeName_vl = os.path.join(args.expt_directory, "log_val.txt")
    file_tr = open(completeName_tr, "w")
    file_val = open(completeName_vl, "w")
    n_steps = 0
    while epochs_completed < number_epochs:

        # Set the gradients to zero
        grad_Weights_final, grad_Biases_final = make_gradients_zero(n_hidden_layers, size_hidden)
        n_dpts = 0
        loss_validation = 0
        for X, Y in zip(X_train, Y_train):
            n_dpts = n_dpts + 1
            loss, a, h, y_pred = forward_propogation(X, Y, w_look_ahead, b_look_ahead, act_function, loss_function, n_hidden_layers)
            #loss_training += loss
            grad_Weights, grad_Biases = back_propagation(loss_function, act_function, loss, a, h, y_pred, w_look_ahead, X, Y,  n_hidden_layers)


            # Update the gradients of weights and biases
            for k in range(1, n_hidden_layers + 2):
                grad_w_lookahead[k] += grad_Weights[k]
                grad_b_lookahead[k] += grad_Biases[k]

            if n_dpts % mini_batch_size == 0:

                n_steps = n_steps + 1
                # Computing the new updates for the weights and biases
                for i in range(1, n_hidden_layers + 2):
                    update_new_weights[i] = momentum * update_old_weights[i] + lr * grad_w_lookahead[i]
                    update_new_biases[i] = momentum * update_old_biases[i] + lr * grad_b_lookahead[i]


                    # Assignment of new updates to old updates
                    update_old_weights[i] = update_new_weights[i]
                    update_old_biases[i] = update_new_biases[i]

                for i in range(1, n_hidden_layers + 2):
                    w_0[i] = w_0[i] - update_new_weights[i]
                    b_0[i] = b_0[i] - update_new_biases[i]

                # Update the look ahead weights and biases used for the next iteration

                for k in range(1, n_hidden_layers + 2):
                    w_look_ahead[k] = w_0[k] - momentum * update_old_weights[k]
                    b_look_ahead[k] = b_0[k] - momentum * update_old_biases[k]

                    # Set the gradients to zero
                grad_Weights_final, grad_Biases_final = make_gradients_zero(n_hidden_layers, size_hidden)

            # Creating the log files for the train and test data
                if n_steps % 100 == 0 and n_steps != 0:

                    error_train = 0 # Training error (accuracy)
                    error_validation = 0 # Validation error (accuracy)
                    y_train = []
                    y_val = []

                # Getting train and validation accuracy
                    for k in X_train:
                        y_train.append(get_final_predictions(k, w_0, b_0, act_function, n_hidden_layers))
                    error_train = np.sum(y_train == Y_train)/len(Y_valid)

                    for k in X_valid:
                        y_val.append(get_final_predictions(k, w_0, b_0, act_function, n_hidden_layers))
                    error_validation = np.sum(y_val == Y_valid)/len(Y_valid)

                # Getting the train and validation losses

                    for X1, Y1 in zip(X_valid, Y_valid):

                        loss1, a1, h1, y_pred1 = forward_propogation(X1, Y1, w_0, b_0, act_function, loss_function, n_hidden_layers)
                        loss_validation_perstep += loss1
                    for X2, Y2 in zip(X_train, Y_train):

                        loss2, a2, h2, y_pred2 = forward_propogation(X2, Y2, w_0, b_0, act_function, loss_function, n_hidden_layers)
                        loss_training_perstep += loss2

                    file_tr.write("Epoch : %f \t" % epochs_completed + "Step : %f \t" % n_steps + "Loss : %.2f \t" %loss_training_perstep + "Error : %.2f \t" %error_train + "lr : %f \n " %lr)

                    file_val.write("Epoch : %f \t" % epochs_completed + "Step : %f \t" % n_steps + "Loss : %.2f \t" %loss_validation_perstep + "Error : %.2f \t" %error_validation + "lr : %f \n " %lr)

                    loss_training_perstep = 0
                    loss_validation_perstep = 0

         # Computing the validation loss after each epoch
        for X3, Y3 in zip(X_valid, Y_valid):

            loss3, a3, h3, y_pred3 = forward_propogation(X3, Y3, w_0, b_0, act_function, loss_function, n_hidden_layers)
            loss_validation += loss3

        error_val.append(loss_validation)
        epochs_completed += 1
        Running_Weights.append(w_0)
        Running_Biases.append(b_0)
        Running_oldupdate_weights.append(update_old_weights)
        Running_oldupdate_biases.append(update_old_biases)
        Running_lookahead_weights.append(w_look_ahead)
        Running_lookahead_biases.append(b_look_ahead)

        # Saving the weights and biases in pickled form after each epoch
        list_of_parameters = []
        for t in range(1, n_hidden_layers + 2):
            list_of_parameters.append(w_0[t])

        for t in range(1, n_hidden_layers + 2):
            list_of_parameters.append(b_0[t])

        save_weights(list_of_parameters, epochs_completed)

        # Annealing of the learning rate
        if anneal == "True":
            if epochs_completed > 1:
                if error_val[epochs_completed - 1] > error_val[epochs_completed - 2]:
                    error_val = error_val[:-1]
                    epochs_completed -=1
                    lr = lr*0.5

                    # Assign the weights and biases to the values in the previous epoch
                    w_0 = Running_Weights[epochs_completed]
                    b_0 = Running_Biases[epochs_completed]
                    update_old_weights = Running_oldupdate_weights[epochs_completed - 1]
                    update_old_biases = Running_oldupdate_biases[epochs_completed - 1]
                    w_look_ahead = Running_lookahead_weights[epochs_completed - 1]
                    b_look_ahead = Running_lookahead_biases[epochs_completed - 1]

    file_tr.close()
    file_val.close()
    return w_0, b_0


# Does Adam
#  CORRECT ONE SEND BY IYER

def adam(X_train, Y_train, X_valid, Y_valid, n_hidden_layers, size_hidden, number_epochs, mini_batch_size, lr, act_function, loss_function, anneal, init, beta1, beta2):
# Initialize the weights and biases
    if args.pre_train == "True" and args.testing_ == "False":

        learned_params = load_weights(args.n_state)
        w_0 = dict()
        b_0 = dict()

        for tt in range(0, n_hidden_layers + 1):
            w_0[tt] = learned_params[tt]

        for tt in range(n_hidden_layers + 1, len(learned_params) ):
            b_0[tt] = learned_params[tt]


    else:
        w_0, b_0 = init_weights_biases(n_hidden_layers, size_hidden, init)
    error_val = [] # Keeps a track of the validation error at every epoch
    epochs_completed = 0
    epsilon = pow(10, -8)

    Running_Weights = [] # Keeps a track of the weights at each epoch
    Running_Weights.append(w_0)
    Running_Biases = []  # Keeps a track of the biases at each epoch
    Running_Biases.append(b_0)
    update_old_weights, update_old_biases = make_gradients_zero(n_hidden_layers, size_hidden)
    update_new_weights, update_new_biases = make_gradients_zero(n_hidden_layers, size_hidden)
    Running_oldupdate_weights = []
    Running_oldupdate_weights.append(update_old_weights)
    Running_oldupdate_biases = []
    Running_oldupdate_biases.append(update_old_biases)
    v_weights_old, v_biases_old = make_gradients_zero(n_hidden_layers, size_hidden)
    v_weights_new, v_biases_new = make_gradients_zero(n_hidden_layers, size_hidden)
    ad_lr_w = []
    ad_lr_b = []
    ad_lr_w.append(v_weights_old)
    ad_lr_b.append(v_biases_old)
    beta_1 = beta1
    beta_2 = beta2
    n_update = 1
    epochs_completed = 0
    loss_training_perstep = 0
    loss_validation_perstep = 0

    # Initiating the log train and log validation files
    completeName_tr = os.path.join(args.expt_directory, "log_train.txt")
    completeName_vl = os.path.join(args.expt_directory, "log_val.txt")
    file_tr = open(completeName_tr, "w")
    file_val = open(completeName_vl, "w")
    n_steps = 0
    while epochs_completed < number_epochs:

        pred_labels_val = [] # Storing the labels for validation loss per epoch
        pred_labels_train = [] # Storing the labels for training loss per epoch
        loss_validation = 0 # Keep track of the validation loss per epoch

        # Set the gradients to zero
        grad_Weights_final, grad_Biases_final = make_gradients_zero(n_hidden_layers, size_hidden)
        n_dpts = 0

        for X, Y in zip(X_train, Y_train):

            n_dpts = n_dpts + 1
            loss, a, h, y_pred = forward_propogation(X, Y, Running_Weights[epochs_completed], Running_Biases[epochs_completed], act_function, loss_function, n_hidden_layers)
            #loss_training_perstep += loss
            grad_Weights, grad_Biases = back_propagation(loss_function, act_function, loss, a, h, y_pred, Running_Weights[epochs_completed], X, Y,  n_hidden_layers)


# Update the  gradients of weights and biases
            for k in range(1, n_hidden_layers + 2):
                grad_Weights_final[k] += grad_Weights[k]
                grad_Biases_final[k] += grad_Biases[k]

            if n_dpts % mini_batch_size == 0:
                #print("HII")
                n_steps = n_steps + 1 # Updating the number of steps completed
                #print(n_steps)
                for i in range(1, n_hidden_layers + 2):
                    update_new_weights[i] = (beta_1 * update_old_weights[i] + (1 - beta_1) * grad_Weights_final[i]) / (1 - pow(beta_1, n_update))
                    update_new_biases[i] = (beta_1 * update_old_biases[i] + (1 - beta_1) * grad_Biases_final[i] ) / (1 - pow(beta_1, n_update))


                    # Assignment of new updates to old updates
                    update_old_weights[i] = update_new_weights[i]
                    update_old_biases[i] = update_new_biases[i]

                for i in range(1, n_hidden_layers + 2):
                    v_weights_new[i] = ( beta_2 * v_weights_old[i] + (1 - beta_2) * (grad_Weights_final[i] *grad_Weights_final[i]) ) / (1 - pow(beta_2, n_update))
                    v_biases_new[i] = ( beta_2 * v_biases_old[i] + (1 - beta_2) * (grad_Biases_final[i] * grad_Biases_final[i]) ) / (1 - pow(beta_2, n_update))

                    v_weights_old[i] = v_weights_new[i]
                    v_biases_old[i] = v_biases_new[i]

                for i in range(1, n_hidden_layers + 2):
                    w_0[i] = w_0[i] - (lr/np.sqrt(v_weights_new[i] + epsilon) ) * update_new_weights[i]
                    b_0[i] = b_0[i] - (lr/np.sqrt(v_biases_new[i] + epsilon) ) * update_new_biases[i]

                Running_Weights[epochs_completed] = w_0
                Running_Biases[epochs_completed] = b_0

                n_update += 1

                # Set the gradients to zero
                grad_Weights_final, grad_Biases_final = make_gradients_zero(n_hidden_layers, size_hidden)

             # Creating the log files for the train and test data
                if n_steps % 100 == 0 and n_steps != 0:
                    #print("HII1")
                    error_train = 0 # Training error (accuracy)
                    error_validation = 0 # Validation error (accuracy)
                    y_train = []
                    y_val = []

                # Getting train and validation accuracy
                    for k in X_train:
                        y_train.append(get_final_predictions(k, w_0, b_0, act_function, n_hidden_layers))
                    error_train = np.sum(y_train == Y_train)/len(Y_valid)

                    for k in X_valid:
                        y_val.append(get_final_predictions(k, w_0, b_0, act_function, n_hidden_layers))
                    error_validation = np.sum(y_val == Y_valid)/len(Y_valid)

                # Getting the train and validation losses

                    for X1, Y1 in zip(X_valid, Y_valid):

                        loss1, a1, h1, y_pred1 = forward_propogation(X1, Y1, w_0, b_0, act_function, loss_function, n_hidden_layers)
                        loss_validation_perstep += loss1

                    for X2, Y2 in zip(X_train, Y_train):

                        loss2, a2, h2, y_pred2 = forward_propogation(X2, Y2, w_0, b_0, act_function, loss_function, n_hidden_layers)
                        loss_training_perstep += loss1

                    file_tr.write("Epoch : %f \t" % epochs_completed + "Step : %f \t" % n_steps + "Loss : %.2f \t" %loss_training_perstep + "Error : %.2f \t" %error_train + "lr : %f \n" %lr)

                    file_val.write("Epoch : %f \t" % epochs_completed + "Step : %f \t" % n_steps + "Loss : %.2f \t" %loss_validation_perstep + "Error : %.2f \t" %error_validation + "lr : %f \n " %lr)

                    loss_training_perstep = 0
                    loss_validation_perstep = 0

    # Computing the validation loss after each epoch
        for X3, Y3 in zip(X_valid, Y_valid):

            loss3, a3, h3, y_pred3 = forward_propogation(X3, Y3, w_0, b_0, act_function, loss_function, n_hidden_layers)
            loss_validation += loss3

        error_val.append(loss_validation)
        epochs_completed += 1
        Running_Weights.append(w_0)
        Running_Biases.append(b_0)
        Running_oldupdate_weights.append(update_old_weights)
        Running_oldupdate_biases.append(update_old_biases)
        ad_lr_w.append(v_weights_new)
        ad_lr_b.append(v_biases_new)

        # Saving the weights and biases in pickled form after each epoch
        list_of_parameters = []
        for t in range(1, n_hidden_layers + 2):
            list_of_parameters.append(w_0[t])

        for t in range(1, n_hidden_layers + 2):
            list_of_parameters.append(b_0[t])

        save_weights(list_of_parameters, epochs_completed)

        # Annealing of the learning rate
        #print(epochs_completed)
        if anneal == "True":
            if epochs_completed > 1:
                if error_val[epochs_completed - 1] > error_val[epochs_completed - 2]:
                    error_val = error_val[:-1] # To rewrite the validation loss entry for the repeated epoch
                    epochs_completed -=1
                    lr = lr * 0.5

                    # Assign the weights and biases to the values in the previous epoch
                    w_0 = Running_Weights[epochs_completed]
                    b_0 = Running_Biases[epochs_completed]
                    update_old_weights = Running_oldupdate_weights[epochs_completed]
                    update_old_biases = Running_oldupdate_biases[epochs_completed]
                    v_weights_old = ad_lr_w[epochs_completed]
                    v_biases_old = ad_lr_b[epochs_completed]

    #print(n_steps)
    #print(n_dpts)
    #print(mini_batch_size)
    file_tr.close()
    file_val.close()
    return w_0, b_0


# Used to obtain the final prediction on the test data
def get_final_predictions(X, Weights, Biases, act_function, n_hidden_layers):
    a = dict()
    h = dict()
    X = np.reshape(X, (100, 1))
    a_old = X

# Computing the pre-activation and activation for hidden layers
    for j in range(1, n_hidden_layers + 1):
        a_new = Biases[j] + np.matmul(Weights[j], a_old)
        a[j] = a_new
        if act_function  == "sigmoid":
            h_ = sigmoid(a_new)
        else:
            h_ = tanh(a_new)
        a_old = h_
        h[j] = h_

#Separate compution of the preactivation and activation for the outut layer
    a[n_hidden_layers + 1] = Biases[n_hidden_layers + 1] + np.matmul(Weights[n_hidden_layers + 1], h[n_hidden_layers])
# The maximum value of the output preactivation layer is subtracted from each element of the preactivation layer to prevent overflow in softmax
    y_pred = np.exp(a[n_hidden_layers + 1] - np.max(a[n_hidden_layers + 1]))/(np.sum(np.exp(a[n_hidden_layers + 1] - np.max(a[n_hidden_layers + 1]))))

    label = np.argmax(y_pred)
    return label

#---------------------------------------------------------------------------------------------------------------------------------------
# The master function
train_dat = pd.read_csv(args.train_dataset)
validation_dat = pd.read_csv(args.validation_dataset)
test_dat = pd.read_csv(args.test_dataset)

X_train, Y_train = train_dat.iloc[1:100, 1:101].values, train_dat.iloc[1:100, 785].values
X_valid, Y_valid = validation_dat.iloc[1:100, 1:101].values, validation_dat.iloc[1:100, 785].values
X_test = test_dat.iloc[1:100, 1:101].values

# Scaling the data
sl = StandardScaler()
X_train = sl.fit_transform(X_train)
X_valid = sl.fit_transform(X_valid)
X_test = sl.fit_transform(X_test)

# Performing PCA on the scaled data
pca = PCA(n_components = 100)
pca.fit(X_train)
X_train = pca.transform(X_train)

X_valid = pca.transform(X_valid)

X_test = pca.transform(X_test)

if args.testing_ == "True" and args.pre_train == "True":
    learned_params = load_weights(int(args.n_state))
    Weights_lrnd = dict()
    Biases_lrnd = dict()

    for tt in range(0, int(args.n_hidden_layers) + 1):
        Weights_lrnd[tt + 1] = learned_params[tt]

    for tt in range(int(args.n_hidden_layers) + 1, len(learned_params) ):
        Biases_lrnd[tt - int(args.n_hidden_layers) ] = learned_params[tt]


# Computing the final loss and predictions
    final_test_predictions = []
    for k in X_test:
        final_test_predictions.append(get_final_predictions(k, Weights_lrnd, Biases_lrnd, args.act_function, int(args.n_hidden_layers)))

    id_ = []
    for u in range(len(final_test_predictions)):
        id_.append(u)

    id_ = pd.DataFrame(id_)
    predict_ = pd.DataFrame(final_test_predictions)

    final = pd.concat([id_, predict_], axis = 1)
    final.columns = ['id', 'label']

    name = args.expt_directory +'\predictions_{}.csv'.format(int(args.n_state))
    final.to_csv(name, index = False)

else:

    if args.opt_algorithm == "gd":

        weights_final, biases_final = vanilla_gradient_descent(X_train, Y_train, X_valid, Y_valid, args.n_hidden_layers, args.size_hidden, args.n_epochs, args.n_batch_size, args.lrng_rate,  args.act_function, args.loss_function, args.anneal_,  init = 0.04)

    if args.opt_algorithm == "momentum":

        weights_final, biases_final = momentum_gradient_descent(X_train, Y_train, X_valid, Y_valid, args.n_hidden_layers, args.size_hidden, args.n_epochs, args.n_batch_size, args.lrng_rate, args.moment,  args.act_function, args.loss_function, args.anneal_ , init = 0.04)

    if args.opt_algorithm == "nag":

        weights_final, biases_final = nestorov_accelerated_gradient_descent(X_train, Y_train, X_valid, Y_valid, args.n_hidden_layers, args.size_hidden, args.n_epochs, args.n_batch_size, args.lrng_rate, args.moment,  args.act_function, args.loss_function , args.anneal_, init = 0.04)

    if args.opt_algorithm == "adam":

        weights_final, biases_final = adam(X_train, Y_train, X_valid, Y_valid, int(args.n_hidden_layers), args.size_hidden, int(args.n_epochs), int(args.n_batch_size), float(args.lrng_rate),  args.act_function, args.loss_function, args.anneal_ , init =0.04, beta1 = 0.5, beta2 = 0.5)

# Code that installs the matplotlib package if it is not already installed
import importlib.metadata, subprocess, sys, os
if 'matplotlib' not in {pkg.metadata['Name'] for pkg in importlib.metadata.distributions()}:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

from copy import deepcopy
from datasets import make_circles
from YAAE import Node
from nn import NN, Optimizer

os.environ['TCL_LIBRARY'] = r'C:/Program Files/Python313/tcl/tcl8.6'

def cross_entropy_loss(y_pred, labels):
    loss = labels * (y_pred.log()) + (1. - labels) * ((1. - y_pred).log())
    return loss.sum(keepdims=False) / -labels.shape[0]

def classification_accuracy(y_pred, labels):
    y_pred_class = np.where(y_pred.data<0.5, 0, 1)
    return np.sum(labels.data == y_pred_class.data) / labels.shape[0]

def plot_dataset(X, y):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

def plot_model_and_dataset(model, X, y):
    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.25
    
    # Generate a grid of points with distance h between them.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid.
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    Z = model(Node(Xmesh, requires_grad=False))
    pred_class = Z.data.reshape(xx.shape)

    # Plot the contour and training examples.
    plt.contourf(xx, yy, pred_class)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def train(model, optimizer, inputs, labels, epochs):
    # Task 1
    # Train a model for a set number of epochs.
    # Replace the line below with your code.
    raise NotImplementedError

def test(model, inputs, labels):
    # Task 2
    # Test a model.
    # Replace the line below with your code.
    raise NotImplementedError

def train_and_test(model, training_data, test_data, iterations, epochs, report=False):
    model = deepcopy(model)
    training_X, training_y = training_data
    training_inputs = Node(training_X, requires_grad=False)
    training_labels = Node(training_y[:, np.newaxis], requires_grad=False)
    test_X, test_y = test_data
    test_inputs = Node(test_X, requires_grad=False)
    test_labels = Node(test_y[:, np.newaxis], requires_grad=False)

    # Task 3.1
    # Initial evaluation of loss and accuracy on the training and test sets.
    # Replace the line below with your code.
    raise NotImplementedError

    if report:
        print(f'Epoch 0: training-loss: {training_loss:.3f} | training-acc: {training_acc:.3f} | test-loss: {test_loss:.3f} | test-acc: {test_acc:.3f}')
    training_losses = [training_loss]
    training_accs = [training_acc]
    test_losses = [test_loss]
    test_accs = [test_acc]
    if report:
        plot_model_and_dataset(model, test_X, test_y)

    # Task 3.2
    # Initialise the optimizer.
    # Replace the line below with your code.
    raise NotImplementedError

    for iter_i in range(iterations):

        # Task 3.3
        # Train the model.
        # Evaluate loss and accuracy on the training and test sets.
        # Replace the line below with your code.
        raise NotImplementedError

        if report:
            print(f'Epoch {(iter_i + 1) * epochs}: training-loss: {training_loss:.3f} | training-acc: {training_acc:.3f} | test-loss: {test_loss:.3f} | test-acc: {test_acc:.3f}')
        training_losses.append(training_loss)
        training_accs.append(training_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if report:
            plot_model_and_dataset(model, test_X, test_y)
    if report:
        plt.plot(range(0, iterations * epochs + 1, epochs), training_losses, label='training loss')
        plt.plot(range(0, iterations * epochs + 1, epochs), test_losses, label='test loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        plt.plot(range(0, iterations * epochs + 1, epochs), training_accs, label='training accuracy')
        plt.plot(range(0, iterations * epochs + 1, epochs), test_accs, label='test accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
    return training_losses, training_accs, test_losses, test_accs

# Task 4
# Initialise a wider model and a deeper model.
model_wide = None
model_deep = None

def test_training_set_sizes(sizes, model):
    sizes = sorted(sizes)
    recorded_training_losses = []
    recorded_training_accs = []
    recorded_test_losses = []
    recorded_test_accs = []

    # Task 5.1
    # Create a test set.
    # Replace the line below with your code.
    raise NotImplementedError

    for size in sizes:

        # Task 5.2
        # Create a training set of the specified size.
        # Obtain the training and test losses and accuracy of the model.
        # Replace the line below with your code.
        raise NotImplementedError

        recorded_training_losses.append(training_losses[-1])
        recorded_training_accs.append(training_accs[-1])
        recorded_test_losses.append(test_losses[-1])
        recorded_test_accs.append(test_accs[-1])
    plt.plot(sizes, recorded_training_losses, label='training loss')
    plt.plot(sizes, recorded_test_losses, label='test loss')
    plt.xlabel('training set size')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.plot(sizes, recorded_training_accs, label='training accuracy')
    plt.plot(sizes, recorded_test_accs, label='test accuracy')
    plt.xlabel('training set size')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    model = NN(nin=2, nouts=[8, 1])
    print(model)

    # Tasks 1, 2, & 3 test code
    training_X, training_y = make_circles(noise=0.1)
    test_X, test_y = make_circles(noise=0.1)
    train_and_test(model, (training_X, training_y), (test_X, test_y), 10, 1000, True)

    # Task 4 test code
    print(model_wide)
    train_and_test(model_wide, (training_X, training_y), (test_X, test_y), 10, 1000, True)
    print(model_deep)
    train_and_test(model_deep, (training_X, training_y), (test_X, test_y), 10, 1000, True)

    # Task 5 test code
    test_training_set_sizes(list(range(50, 501, 50)), model)
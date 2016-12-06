import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
import matplotlib.pyplot as plt

def load_data():
    # Load the raw CIFAR-10 data.
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # As a sanity check, we print out the size of the training and test data.
    print 'Training data shape: ', X_train.shape
    print 'Training labels shape: ', y_train.shape
    print 'Test data shape: ', X_test.shape
    print 'Test labels shape: ', y_test.shape

    # Subsample the data for more efficient code execution in this exercise
    num_training = 5000
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # visualize_data(X_train, y_train)

    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print X_train.shape, X_test.shape

    return X_train, y_train, X_test, y_test, num_test

def visualize_data(X_train, y_train):
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

def compute_distances(X_train, y_train, X_test, y_test, num_test):
    classifier.train(X_train, y_train)
    # # Two loops
    # print 'Two loops'
    # dists = classifier.compute_distances_two_loops(X_test)
    # print dists.shape

    # plt.imshow(dists, interpolation='none')
    # plt.show()

    # y_test_pred = classifier.predict_labels(dists, k=1)
    # determine_accuracy(y_test_pred, y_test, num_test)

    # y_test_pred = classifier.predict_labels(dists, k=5)
    # determine_accuracy(y_test_pred, y_test, num_test)

    # # One loop
    # print 'One loop'
    # dists_one = classifier.compute_distances_one_loop(X_test)
    # check_if_the_same(dists, dists_one)

    # No loops
    print 'No loops'
    dists_two = classifier.compute_distances_no_loops(X_test)
    # check_if_the_same(dists, dists_two)
    return dists_two

def determine_accuracy(y_test_pred, y_test, num_test):
    print 'y_test_pred == y_test'
    print y_test_pred == y_test
    print np.sum(y_test_pred == y_test)
    print 'num_test'
    print num_test
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
    return accuracy

def check_if_the_same(dists, dists_compare):
    difference = np.linalg.norm(dists - dists_compare, ord='fro')
    print 'Difference was: %f' % (difference, )
    if difference < 0.001:
        print 'Good! The distance matrices are the same'
    else:
        print 'Uh-oh! The distance matrices are different'

def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

def cross_validation(X_train, y_train):
    print '# 2'
    print classifier.y_train
    print classifier.y_train.shape

    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []
    ################################################################################
    # TODO:                                                                        #
    # Split up the training data into folds. After splitting, X_train_folds and    #
    # y_train_folds should each be lists of length num_folds, where                #
    # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
    # Hint: Look up the numpy array_split function.                                #
    ################################################################################
    X_train_folds = np.split(X_train, num_folds)
    y_train_folds = np.split(y_train, num_folds)
    num_test = X_train_folds[0].shape[0]

    print '# 3'
    print classifier.y_train
    print classifier.y_train.shape
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    k_to_accuracies = {}

    ################################################################################
    # TODO:                                                                        #
    # Perform k-fold cross validation to find the best value of k. For each        #
    # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
    # where in each case you use all but one of the folds as training data and the #
    # last fold as a validation set. Store the accuracies for all fold and all     #
    # values of k in the k_to_accuracies dictionary.                               #
    ################################################################################
    for k in k_choices:
        k_to_accuracies[k] = []

    for i in range(num_folds):
        folds = range(num_folds)
        folds.pop(i)
        X = np.vstack(X_train_folds[fold] for fold in folds)
        y = np.concatenate([y_train_folds[fold] for fold in folds])
        dists = compute_distances(X, y, X_train_folds[i], y_train_folds[i], num_test)
        for k in k_choices:
            y_test_pred = classifier.predict_labels(dists, k)
            accuracy = determine_accuracy(y_test_pred, y_train_folds[i], num_test)
            k_to_accuracies[k].append(accuracy)

    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print 'k = %d, accuracy = %f' % (k, accuracy)

    # plot the raw observations
    for k in k_choices:
      accuracies = k_to_accuracies[k]
      plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

    best_k = 1
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict(X_test, k=best_k)
    determine_accuracy(y_test_pred,y_test, num_test)


X_train, y_train, X_test, y_test, num_test = load_data()
raw_input('Any key to continue...')

classifier = KNearestNeighbor()
compute_distances(X_train, y_train, X_test, y_test, num_test)
print '# 1'
print classifier.y_train
print classifier.y_train.shape
raw_input('Any key to continue...')

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print 'No loop version took %f seconds' % no_loop_time
raw_input('Any key to continue...')

cross_validation(X_train, y_train)





import numpy
import numpy as np
import pickle

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def compute_distances_no_loops(X, X_train):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    print dists.shape
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # X = X.reshape(X.shape[0], 1 ,X.shape[1])
    X = np.expand_dims(X, axis=1)
    diff = X_train - X
    diff_sq = np.square(diff)
    diff_sum = np.sum(diff_sq, axis=2)
    diff_root = np.sqrt(diff_sum)
    dists = diff_root
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
X_train = data_batch_1['data'][:5000]

test_batch = unpickle('cifar-10-batches-py/test_batch')
X = test_batch['data'][:500]

# print X_train.shape
# print X.shape
compute_distances_no_loops(X, X_train)














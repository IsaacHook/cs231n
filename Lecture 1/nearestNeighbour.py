import numpy
import pickle

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data():
    # unpickle training data
    data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
    data_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
    data_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
    data_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
    data_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')

    # save training data and labels
    training_data = numpy.concatenate((data_batch_1['data'], data_batch_2['data'], data_batch_3['data'], data_batch_4['data'], data_batch_5['data']), axis=0)
    training_labels = data_batch_1['labels'] + data_batch_2['labels'] + data_batch_3['labels'] + data_batch_4['labels'] + data_batch_5['labels']

    # unpickle test data
    test_batch = unpickle('cifar-10-batches-py/test_batch')

    # save test data and labels
    test_data = test_batch['data']
    test_labels = test_batch['labels']

    batches_meta = unpickle('cifar-10-batches-py/batches.meta')

    return training_data, training_labels, test_data, test_labels

class neareastNeighbour:
    def __init__(self):
        pass

    def train(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

    def predict(self, test_data):
        test_predict = []

        for row in test_data:
            distances = numpy.sum(numpy.abs(self.training_data - row), axis=1)
            nearest = numpy.argmin(distances)
            test_predict.append(self.training_labels[nearest])

        return test_predict 

training_data, training_labels, test_data, test_labels = load_data()
nn = neareastNeighbour()
nn.train(training_data, training_labels)
test_predict = nn.predict(test_data)
print "accuracy: %d " + np.mean(test_predict==test_labels)












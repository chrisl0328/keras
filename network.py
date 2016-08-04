#!/usr/bin/env/ python3

from argparse import ArgumentDefaultsHelpFormatter, ArgumentTypeError, ArgumentParser
from math import sqrt
import numpy as np
import theano
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from fuel.datasets.hdf5 import H5PYDataset
import h5py
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import ForceFloatX



#TODOlater: description
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('ccgbank', type=str, metavar='CCGBANKHDF5',
                    help='CCGbank hdf5 file')
# TODOlater: tune batch size
parser.add_argument('-t', '--batch-size', type=int, default=128,
                    metavar='INT', help='minibatch size')

training_params = parser.add_argument_group('training hyperparameters')
# TODOlater: tune learning rate
training_params.add_argument('-a', '--alpha', type=float, default=0.1,
                             metavar='FLOAT', help='learning rate')
# TODOlater: tune regularization coefficient
training_params.add_argument('-b', '--beta', type=float, default=0.001,
                             metavar='FLOAT',
                             help='L2 regularization coefficient')
# TODOlater: tune dropout probability (plus different probs for different layers)
training_params.add_argument('-d', '--dropout', type=float, default=0.5,
                             metavar='FLOAT', help='dropout probability')
# TODOlater: tune momentum
# TODOlater: use momentum ~0.95 or 0.99 if using dropout
training_params.add_argument('-g', '--gamma', type=float, default=0.9,
                             metavar='FLOAT', help='momentum coefficient')

model_params = parser.add_argument_group('model parameters')
# TODOlater: tune filter size
model_params.add_argument('-f', '--filter-size', type=int, nargs=2,
                          default=[5, 7], metavar='INT',
                          help='filter size; first dimension is embedding '
                               'filter width and second is temporal')
# TODOlater: tune feature maps
model_params.add_argument('-m', '--feature-maps', type=int, default=1,
                          metavar='INT', help='number of feature maps')
# TODOlater: tune wide conv + narrowing pool vs. fixed-width all the way
model_params.add_argument('-w', '--fixed-width', action='store_false',
                          dest='wide',
                          help='make convolution output length the same as '
                               'input')
# TODOlater: tune pooling size
model_params.add_argument('-p', '--pooling-size', type=int, nargs=2,
                          default=[10, 3], metavar='INT',
                          help='pooling size; first dimension is embedding '
                               'pooling width and second is temporal')
# TODOlater: tune MLP hidden layer size
model_params.add_argument('-l', '--hidden-size', type=int, default=200,
                          metavar='INT', help='MLP hidden layer size')

model_io = parser.add_argument_group('model I/O')
model_io.add_argument('-i', '--load-model', type=str, metavar='FILE',
                      help='load model and resume trainng from given file')
model_io.add_argument('-o', '--model-out', type=str, metavar='FILE',
                      default='best.model.pkl',
                      help='file to which the best (dev) model should be '
                           'saved')

stopping = parser.add_argument_group('stopping criteria')
stopping.add_argument('--max-epochs', type=int, default=100, metavar='INT',
                      help='maximum number of epochs to train; 0 means no '
                           'limit')
stopping.add_argument('--min-epochs', type=int, default=10, metavar='INT',
                      help='minimum number of epochs to train after finding '
                           'best CCE')
stopping.add_argument('--threshold', dest='thresh', type=float, default=0.01,
                      metavar='FLOAT',
                      help='minimum CCE improvement threshold')
# I can't think of any good reason to allow the seed to be changed---just pick
# a value and stick with it (1 is a sensible default)
# parser.add_argument('--seed', type=int, default=1, metavar=INT,
#                     help='PRNG seed')

args = parser.parse_args()

if not args.wide:
    if args.filter_size[1] % 2 == 0:
        raise ArgumentTypeError('temporal filter width must be odd in the '
                                'fixed-width convolution case')
if args.filter_size[0] <= 0:
    raise ArgumentTypeError('embedding filter width must be at least 1')
if args.filter_size[1] < 2:
    raise ArgumentTypeError('temporal filter width must be at least 2')

args.filter_size = tuple(args.filter_size)

if args.wide:
    # this doesn't, strictly speaking, *have* to be the case, as we could just
    # pad as necessary, but let's just leave it as such for now; pooling
    # padding doesn't make a whole lot of sense (AFAICT) anyway
    # i do think that the pooling size has to be at least the filter size,
    # otherwise you will definitely end up with a larger output
    args.pooling_size[1] = args.filter_size[1]
else:
    if args.pooling_size[1] % 2 == 0:
        raise ArgumentTypeError('temporal pooling width must be odd in the '
                                'fixed-width convolution case')
    if args.pooling_size[1] < 3:
        raise ArgumentTypeError('temporal pooling width must be at least 3')
if args.pooling_size[0] < 2:
    raise ArgumentTypeError('embedding pooling width must be at least 2')

if args.feature_maps < 1:
    raise ArgumentTypeError('number of feature maps must be at least 1')
if not 0 < args.alpha < 1:
    raise ArgumentTypeError('learning rate must be between 0 and 1 exclusive')
if not 0 <= args.beta < 1:
    raise ArgumentTypeError('L2 regularization coefficient must be at least 0 '
                            'and less than 1')
if not 0 <= args.gamma < 1:
    raise ArgumentTypeError('momentum coefficient must be at least 0 and less '
                            'than 1')
if not 0 <= args.dropout < 1:
    raise ArgumentTypeError('dropout probability must be at least 0 and less '
                            'than 1')
if args.batch_size < 1:
    raise ArgumentTypeError('batch size must be at least 1')
if args.thresh < 0:
    raise ArgumentTypeError('CCE threshold must be at least 0')
if args.max_epochs < 0:
    raise ArgumentTypeError('maximum epoch count must be at least 0')
if args.max_epochs < 1:
    raise ArgumentTypeError('minimum epoch count must be at least 1')

try:
    with h5py.File(args.ccgbank, mode='r') as ccgbank:
        n_cat = ccgbank.attrs['num_categories']
        embdim = ccgbank.attrs['embedding_length']
except OSError as e:
    msg = "couldn't open {}: {}".format(args.ccgbank, e)
    raise ArgumentTypeError(msg) from None

if args.load_model:
    try:
        f = open(args.load_model, 'r')
        f.close()
    except OSError as e:
        msg = "couldn't open {}: {}".format(args.load_model, e)
    raise ArgumentTypeError(msg) from None


#start
train_vectors = H5PYDataset('glove.hdf5', which_sets = ('train',), load_in_memory = True)
print(type(train_vectors))
print(train_vectors)
#print(train_vectors.num_examples)
train_scheme = SequentialScheme(train_vectors.num_examples, 1)
#print(scheme)
train_stream = DataStream(train_vectors, iteration_scheme = train_scheme)
#print(stream.get_epoch_iterator())


dev_vectors = H5PYDataset('glove.hdf5', which_sets = ('dev',), load_in_memory = True)
dev_scheme = SequentialScheme(dev_vectors.num_examples, 1)
dev_stream = DataStream(dev_vectors, iteration_scheme = dev_scheme)

test_vectors = H5PYDataset('glove.hdf5', which_sets = ('test',), load_in_memory = True)
test_scheme = SequentialScheme(test_vectors.num_examples, 1)
test_stream = DataStream(test_vectors, iteration_scheme = test_scheme)


count = 0
for data in train_stream.get_epoch_iterator():
    if count == 1:
        break
    count += 1
    print(type(data[1]))
    print(len(data))
    print(data)

train_data = train_vectors.data_sources
print(type(train_data))
#print(train_data)
train_arraydata = np.array(train_data)
print(type(train_arraydata))
print(train_arraydata)

print(train_arraydata.shape)

#change data to proper one
'''
np.random.seed(1000)
batch_size = 128
nb_classes = 10
nb_epoch = 20

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print(type(X_train))
#print(X_train)
#print(type(y_train))
#print(y_train)
#print(type(mnist.load_data))
#print(mnist.load_data)

X_train = X_train.reshape(60000, 784)
#print(type(X_train))
#print(X_train)
X_train = X_train.astype('float32')
#print(type(X_train))
#print(X_train)
X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')

X_train /= 255
#print(type(X_train))
#print(X_train)
X_test /= 255

#print(X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
#print(type(Y_train))
#print(Y_train)
Y_test = np_utils.to_categorical(y_test, nb_classes)
'''
#layers
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
#print(type(model))
#print(model)

#how does batch size work?
model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 1, validation_data = (X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose = 0)
#print(score)

#print('Test score: ', score[0])
#print('Test accuracy: ', score[1]) 


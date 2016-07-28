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

train_vectors = H5PYDataset('glove.hdf5', which_sets = ('train',), load_in_memory = True)
print(train_vectors.num_examples)

scheme = SequentialScheme(train_vectors.num_examples, 5)
print(scheme)

stream = DataStream(train_vectors, iteration_scheme = scheme)
print(stream.get_epoch_iterator())

count = 0
for data in stream.get_epoch_iterator():
    if count == 1:
        break
    count += 1
    print(data)

data = train_vectors.data_sources
print(type(data))
arraydata = np.array(data)

print(type(arraydata))
print(arraydata.shape)

np.random.seed(1000)
batch_size = 128
nb_classes = 10
nb_epoch = 20

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_train = X_train.astype('float32')
X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

history = model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 1, validation_data = (X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose = 0)
print('Test score: ', score[0])
print('Test accuracy: ', score[1]) 


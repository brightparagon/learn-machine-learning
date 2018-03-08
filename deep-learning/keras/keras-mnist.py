from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

# read mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize data
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float')
X_train /= 255
X_test /= 255

# convert label data to array of zero and one
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# define model structure
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# build the model
model.compile(
  loss='categorical_crossentropy',
  optimizer=Adam(),
  metrics=['accuracy']
)

# train
hist = model.fit(X_train, y_train)

# evaluate
score = model.evaluate(X_test, y_test, verbose=1)
print("loss=", score[0])
print("accuracy=", score[1])

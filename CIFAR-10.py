import tensorflow
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import LearningRateScheduler


def learning_rate(epoch):
  lrate = 0.001
  if epoch > 75:
    lrate = 0.0005
  if epoch > 100:
    lrate = 0.0003
  return lrate

num_classes = 10
batch_size = 128
weighting = 1e-4
epsilon = 1e-5

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

print(x_train.shape[0], "No. of training samples")
print(x_test.shape[0], 'No. of test samples')


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(weighting),
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization(epsilon=epsilon, momentum=.1))
model.add((Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(weighting))))
model.add(Activation('relu'))
model.add(BatchNormalization(epsilon=epsilon, momentum=.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(weighting)))
model.add(Activation('relu'))
model.add(BatchNormalization(epsilon=epsilon, momentum=.1))
model.add((Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(weighting))))
model.add(Activation('relu'))
model.add(BatchNormalization(epsilon=epsilon, momentum=.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(weighting)))
model.add(Activation('relu'))
model.add(BatchNormalization(epsilon=epsilon, momentum=.1))
model.add((Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(weighting))))
model.add(Activation('relu'))
model.add(BatchNormalization(epsilon=epsilon, momentum=.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(weighting)))
model.add(Activation('relu'))
model.add(BatchNormalization(epsilon=epsilon, momentum=.1))
model.add((Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(weighting))))
model.add(Activation('relu'))
model.add(BatchNormalization(epsilon=epsilon, momentum=.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

datagen.fit(x_train)
optimizer = optimizers.rmsprop(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=64), steps_per_epoch=(50000. / 64), epochs=125,
                    verbose=1, validation_data=(x_test, y_test), callbacks=[LearningRateScheduler(learning_rate)])

#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100, scores[0]))
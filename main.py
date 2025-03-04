from __future__ import print_function
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping

# Set random seed for reproducibility
np.random.seed(42)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


def build_model(num_hidden_layers=2, neurons_per_layer=512, activation='relu',
                optimizer='RMSprop', learning_rate=0.001, dropout_rate=0.2,
                batch_norm=False):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation=activation, input_shape=(784,)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for _ in range(num_hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(10, activation='softmax'))

    if optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = SGD(learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def plot_config(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


config = {
    'num_hidden_layers': 2,
    'neurons_per_layer': 500,
    'activation': 'sigmoid',
    'optimizer': 'Adam',
    'learning_rate': 0.00095,
    'dropout_rate': 0.3,
    'batch_norm': True
}




# Build, train, and evaluate the model
model = build_model(**config)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
history = model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1,
                    validation_data=(x_test, y_test), callbacks=[early_stopping])
plot_config(history)

# Save model
model.save('mnist_model.h5')

# Load model
loaded_model = keras.models.load_model('mnist_model.h5')

# Evaluate model
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)  # warning 출력 방지
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

class KerasDNN:
    def __init__(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessed_data.get_preprocessed_data(is_for_nn=True)

    def make_keras_model(self, n_li, activation_li, r_seed=0, lr=0.01,
                  loss="categorical_crossentropy", metrics=["accuracy"]):
        input_dim = self.X_train.shape[1]
        output_dim = self.y_train.shape[1]

        np.random.seed(r_seed)
        self.model = Sequential()

        n_li.append(output_dim)

        for idx, n in enumerate(n_li):
            if idx == 0:
                self.model.add(Dense(120, input_dim=input_dim, activation=activation_li[idx]))
            else:
                self.model.add(Dense(n, activation=activation_li[idx]))
        self.model.compile(optimizer=SGD(lr=lr), loss=loss, metrics=metrics)
        
        return self.model
    
    def run_keras_model(self, epochs=1000, batch_size=100):
        self.hist = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                   validation_data=(self.X_test, self.y_test), verbose=2)
import numpy as np
import pandas as pd
import tensorflow as tf

class DNNModel:
    def __init__(self, sess, name, preprocessed_data, lr=0.01, nn=0):
        self.sess = sess
        self.name = name
        self.preprocessed_data = preprocessed_data
        X, y = preprocessed_data.get_original_data()
        if self.preprocessed_data.is_vectorize:
            X = preprocessed_data.get_vectorized_data()
        self.features_num = X.shape[1]
        self.class_num = len(y.unique())
        self.neuron_num = nn
        if nn == 0:
            self.neuron_num = 20 if preprocessed_data.is_vectorize else 120
            
        self.lr = lr
        self.__build_net()
        
    def __build_net(self):
        with tf.variable_scope(self.name):
            
            self.X = tf.placeholder(tf.float32, shape=[None, self.features_num])
            self.y = tf.placeholder(tf.float32, shape=[None, self.class_num])

            # layer 1
            W1 = tf.get_variable("W1", shape=[self.features_num, self.neuron_num], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([self.neuron_num]))
            layer1 = tf.nn.relu(tf.matmul(self.X, W1) + b1)

            # layer 2
            W2 = tf.get_variable("W2", shape=[self.neuron_num, self.neuron_num], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([self.neuron_num]))
            layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

            # layer 3
            W3 = tf.get_variable("W3", shape=[self.neuron_num, self.neuron_num], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([self.neuron_num]))
            layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

            # layer 4
            W4 = tf.get_variable("W4", shape=[self.neuron_num, self.class_num], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([self.class_num]))
            self.logits = tf.matmul(layer3, W4) + b4
            
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    
    def predict(self, X_test=None, keep_prob=0.7, training=False):
        if X_test == None:
            X_test = self.X_test
        return self.sess.run(self.logits, feed_dict={self.X : X_test})
        
    def get_accuracy(self, keep_prob=0.7, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X : self.X_test, self.y : self.y_test})
    
    def train(self, keep_prob=0.7, training=True):
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessed_data.get_preprocessed_data(is_for_nn=True) 
        return self.sess.run([self.cost, self.optimizer], 
                             feed_dict={self.X: self.X_train, self.y: self.y_train})
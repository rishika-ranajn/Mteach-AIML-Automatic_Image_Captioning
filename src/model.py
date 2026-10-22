import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
import time
import json
import random
from PIL import Image
from IPython.display import display
class BahdanauAttention(tf.keras.Model):
    def __init__(self,units):
        super(BahdanauAttention, self).__init__()
        self.w1=tf.keras.layers.Dense(units)
        self.w2=tf.keras.layers.Dense(units)
        self.v=tf.keras.layers.Dense(1)
    def call(self,features,hidden):
        hidden_with_time_axis=tf.expand_dims(hidden, 1)
        attention_hidden_layer=(tf.nn.tanh(self.w1(features)+self.w2(hidden_with_time_axis)))
        score=self.v(attention_hidden_layer)
        attention_weights=tf.nn.softmax(score,axis=1)
        content_vector=attention_weights*features
        content_vector=tf.reduce_sum(content_vector,axis=1)
        return content_vector, attention_weights
class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc=tf.keras.layers.Dense(embedding_dim)
    def call(self,x):
        x=self.fc(x)
        x=tf.nn.relu(x)
        return x
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units=units
        self.embedding=tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru=tf.keras.layers.GRU(self.units,return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1=tf.keras.layers.Dense(self.units)
        self.fc2=tf.keras.layers.Dense(vocab_size)
        self.attention=BahdanauAttention(self.units)
    def call(self,x,features,hidden):
        context_vector, attention_weights=self.attention(features,hidden)
        x=self.embedding(x)
        x=tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state= self.gru(x,initial_state=hidden)
        x=self.fc1(output)
        x=tf.reshape(x,(-1,x.shape[2]))
        x=self.fc2(x)
        return x, state, attention_weights
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

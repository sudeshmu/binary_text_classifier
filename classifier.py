#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:48:14 2020

@author: sudeshmu
"""
import pandas as pd
import pickle as pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import LSTM, Embedding, Input, Activation, Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import Model
from keras.callbacks import EarlyStopping

dataset_file_path = "spam.csv"

max_words = 1000
max_len = 150
train_test_ratio = 0.8
train_batch_size=128
n_epochs=10

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

def train_n_test ():
    data = pd.read_csv (dataset_file_path, delimiter=',', encoding='latin-1')
    X = data.v2
    y = data.v1
    
    label_encoder = LabelEncoder ()
    y = label_encoder.fit_transform (y)
    
    X_train, X_test, y_train, y_test = train_test_split (X, y, 
                                                         train_size=train_test_ratio, 
                                                         shuffle=True)
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    
    model = RNN()
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    
    model.fit(sequences_matrix,y_train,batch_size=train_batch_size,epochs=n_epochs,
              validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    
    print ("Training Completed")
    
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    
    accr = model.evaluate(test_sequences_matrix,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    

train_n_test ()

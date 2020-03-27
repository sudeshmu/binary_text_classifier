# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
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

dataset_file_path = "/home/sudeshmu/work/kaggle/sms-spam-collection-dataset/spam.csv"

label_encoder_path = "label_encoder"
tokenizer_sequences_path = "tokenizer_sequences"
model_path = "model_path"

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

def serialize_object (label_encoder, sequences, model):
    pickle.dump (label_encoder, open (label_encoder_path, 'ab'))
    pickle.dump (sequences, open (tokenizer_sequences_path, 'ab'))
    pickle.dump (model,  open (model_path, 'ab'))

def deserialize_object ():
    return (pickle.load (open (label_encoder_path, 'rb')),
            pickle.load (open (tokenizer_sequences_path, 'rb')),
            pickle.load (open (model_path, 'rb')))

def train ():
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
    
    serialize_object (label_encoder, sequences, model)
    print ("Training Completed")
    
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    
    accr = model.evaluate(test_sequences_matrix,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    

def test (test_str_val):
    (label_encoder, tokenizer_sequences, model) = deserialize_object() 
    test_vec = tokenizer_sequences.pad_sequences (test_str_val,maxlen=max_len)
    prediction = model.predict (test_vec)
    return label_encoder.inverse_transform (prediction)
    
train ()

test_str = "promotion offer message"
print ("Prediction for text = '%s' is %s".format (test_str, test(test_str)))

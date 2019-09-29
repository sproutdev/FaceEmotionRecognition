#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import fer2013
from csv import reader

with open('C:\\Users\\abhi\\HACK UTA\\fer2013\\fer2013.csv', 'r') as f:
    data = list(reader(f)) #Imports the CSV

x_train = []
y = []
for i in data:
    y.append(int(i[0]))
    a = i[1].split(' ')
    x_train.append([int(i) for i in a])

x_train = np.asarray(x_train)
y = np.asarray(y)

x_train = x_train/255.0
print(x_train[0:3])

model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(7, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y, epochs = 5, validation_split = 0.15)


# In[ ]:





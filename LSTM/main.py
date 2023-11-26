import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
from nltk.tokenize  import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.layers import Embedding
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.convolutional import MaxPooling1D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.core import Flatten
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy , CategoricalCrossentropy


from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import warnings

from tensorflow.python.keras.utils.np_utils import to_categorical

warnings.filterwarnings("ignore")

df = pd.read_csv("../Materias/Twitter_Data.csv")

df['category'].mask(df['category'] == -1,'negative',  inplace=True)
df['category'].mask(df['category'] == 0,'neutral',  inplace=True)
df['category'].mask(df['category'] == 1,'positive',  inplace=True)

print(df.head())

df= df.dropna()
df.duplicated().sum()
print(df['category'].value_counts())

fig = plt.figure(figsize=(7, 5))
sns.countplot(x="category", data=df)
plt.title("Dataset labels distribuition")
plt.show()

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_text'])
df['clean_text'] = tokenizer.texts_to_sequences(df['clean_text'])
print(df['clean_text'])

df['category'].mask(df['category'] == 'negative',-1,  inplace=True)
df['category'].mask(df['category'] == 'neutral',0,  inplace=True)
df['category'].mask(df['category'] == 'positive',1,  inplace=True)
df['category']

X_train, X_test, y_train, y_test = train_test_split(df['clean_text'],df['category'], test_size=0.2, random_state=40)
print('X_train:',len(X_train))
print('y_train:',len(y_train))
print('X_test:',len(X_test))
print('y_test:',len(y_test))

X_train = pad_sequences( X_train, maxlen=100 ,dtype='float32')
X_test = pad_sequences( X_test, maxlen=100 ,dtype='float32')
len(tokenizer.index_word)

model = Sequential()
model.add(Embedding(len(tokenizer.index_word) + 1, input_length=100, output_dim=50))
model.add(Bidirectional(LSTM(100)))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile("adam", loss='binary_crossentropy', metrics=["accuracy"])
model.summary()

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss",patience=5,verbose=True)

X_train = pad_sequences(X_train, maxlen=100, dtype='float32')
X_test = pad_sequences(X_test, maxlen=100, dtype='float32')

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

history_1 = model.fit(X_train, y_train, batch_size=64, epochs=7,
                      validation_data=(X_test , y_test), callbacks=[early_stop])

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

model = Sequential()
model.add(Embedding(len(tokenizer.index_word)+1, input_length= 64 ,output_dim =100))
model.add(LSTM(100))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.compile("adam", loss= 'categorical_crossentropy' ,metrics=["accuracy"])
model.summary()

history = model.fit(X_train , y_train ,batch_size=256, epochs=4,
                    validation_data=(X_test , y_test),callbacks=[early_stop])

results = model.evaluate(X_test, y_test, batch_size=64)
print(results)
#%%
import pandas as pd
import numpy as np
from keras.layers import Dense, Flatten, LSTM, GRU, SimpleRNN, Embedding
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import re
# %%
df = pd.read_csv('../datasets/IMDB Dataset.csv')
# %%
df.head(10)
# %%
df.info()
# %%
df.describe()
# %%
df['review'] = df['review'].apply(lambda x: re.sub(r'[,.!&]', '', x))
# %%
df['review']
# %%
tokenizer = Tokenizer()
# %%
input_sequences = np.array(df['review'])
# %%
input_sequences[0]
# %%
tokenizer.fit_on_texts(input_sequences)
# %%
tokenizer.word_index
# %%
vocab_size = len(tokenizer.word_counts) + 1
embeded_dim = 64
max_length = 140
# %%
vocab_size
# %%
train_sequences = tokenizer.texts_to_sequences(input_sequences)
# %%
padded_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
# %%
padded_sequences.shape
# %%
y = np.array(df['review'].apply(lambda x: 1 if x == 'positive' else 0))
# %%
y.shape
# %%
from keras.models import Sequential
# %%
# Normal Embedding Word Model
model1 = Sequential()

model1.add(Embedding(input_dim=vocab_size, output_dim=embeded_dim, input_length=max_length))
model1.add(Flatten())
model1.add(Dense(units=20, activation='relu'))
model1.add(Dense(units=1, activation='sigmoid'))
# %%
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# %%
from sklearn.model_selection import train_test_split
# %%
X_train ,X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=41)
# %%
history_model1 = model1.fit(X_train, y_train, epochs=5)
# %%
model1.evaluate(X_test, y_test)
# %%
# Simple RNN Model
model2 = Sequential()

model2.add(Embedding(input_dim=vocab_size, output_dim = embeded_dim, input_length=max_length))
model2.add(SimpleRNN(units=64))
model2.add(Dense(units=10, activation='relu'))
model2.add(Dense(units=1, activation='sigmoid'))
# %%
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
# %%
model2.fit(X_train, y_train, epochs=4)
# %%
model2.evaluate(X_test, y_test)
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

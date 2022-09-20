#%%
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
# %%
df = pd.read_csv('../datasets/IMDB Dataset.csv')
# %%
df.head(10)
# %%
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
# %%
X = np.array(df['review'])
# %%
y = np.array(df['sentiment'])
# %%
vocab_size = 10000
max_length = 140
embedded_size = 64
# %%
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
# %%
tokenizer.fit_on_texts(X)
# %%
tokenizer.word_index
# %%
X_processed = tokenizer.texts_to_sequences(X)
#%%
X_processed
# %%
X_padded = pad_sequences(X_processed, maxlen=max_length, truncating='post', padding='post')
# %%
X_padded.shape
# %%
X_padded
# %%
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
# %%
model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=embedded_size, input_length=max_length))
model.add(Flatten())
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# %%
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# %%
model.summary()
# %%

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=41)
# %%
X_train, X_val , y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=41)
# %%
history_model = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
# %%
import matplotlib.pyplot as plt

plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'validation'], loc='upper left')
# %%
model_2 = Sequential()

model_2.add(Embedding(input_dim=vocab_size, output_dim=embedded_size, input_length=max_length))
model_2.add(Flatten())
model_2.add(Dense(units=5, activation='relu'))
model_2.add(Dense(units=1, activation='sigmoid'))
# %%
model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_model_2 = model_2.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
# %%
import matplotlib.pyplot as plt

plt.plot(history_model_2.history['accuracy'])
plt.plot(history_model_2.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'validation'], loc='upper left')
# %%
model_2.evaluate(X_test, y_test)
# %%
model.evaluate(X_test, y_test)
# %%
model.save('../models/first_model.h5')
# %%
model_2.save('../models/second_model.h5')
# %%

#%%
import numpy as np
from tokenizer import Tokenizer
tokenizer = Tokenizer()

from keras.layers import Dense
# %%
from keras.models import Sequential
# %%
txt = ['''We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.''']
# %%
tokenizer.fit_on_texts(txt)
# %%
tokenizer.word_index
# %%
tokenizer.one_hot_encoding()
# %%
tokenizer.one_hot_matrix.shape

# %%
X_train = []
y_train = []
# %%
tokenizer.bag_of_words

# %%
tokenizer.word_index
# %%
tokenizer.fit_to_bag(txt)
# %%
tokenizer.bag_of_words
# %%
for context, target in tokenizer.bag_of_words:
    for word in context:
        X_train.append(tokenizer.find_one_hot_vector(word))
        y_train.append(tokenizer.find_one_hot_vector(target))
# %%
X_train = np.array(X_train)
y_train = np.array(y_train)
# %%
X_train.shape

# %%
len(tokenizer.bag_of_words)
# %%
tokenizer.bag_of_words
# %%
y_train
# %%

# %%

# %%

# %%

# %%
X_train.shape
# %%
y_train.shape
# %%
X_train
# %%

# %%

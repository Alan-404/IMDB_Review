#%%
import numpy as np
from tokenizer import Tokenizer
tokenizer = Tokenizer()
#%%
txt = ["I'm about to study the idea of a computational process"]
# %%
tokenizer.fit_to_bag(txt)
# %%
tokenizer.bag_of_words
# %%
tokenizer.word_index
# %%
tokenizer.one_hot_encoding()
#%%
a = tokenizer.find_one_hot_vector('about')
# %%
tokenizer.one_hot_matrix

#%%
tokenizer.bag2onehot()
# %%
tokenizer.bag_of_onehot[0]
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

#%%
bag = []
window_size = 2
# %%

# %%
bag
# %%
tokenizer.fit_on_texts(txt)

# %%
tokenizer.word_index
# %%
tokenizer.index_word
# %%

# %%
len(tokenizer.word_index)
# %%
vocab_size = len(tokenizer.word_index)
embed_dim = 10
# %%
def linear(m, theta):
    w = theta
    return np.dot(m, w)
# %%
def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x/e_x.sum())
# %%
def NLLLoss(logs, targets):
    out = logs[range(len(targets)), targets]
    return -out.sum()/len(out)
# %%
def log_softmax_crossentropy_with_logits(logits,target):

    out = np.zeros_like(logits)
    out[np.arange(len(logits)),target] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- out + softmax) / logits.shape[0]
# %%
def forward(context_idxs, theta, embeddings):
    m = embeddings[context_idxs].reshape(1, -1)
    n = linear(m, theta)
    o = log_softmax(n)
    
    return m, n, o
# %%
def backward(preds, theta, target_idxs):
    m, n, o = preds
    
    dlog = log_softmax_crossentropy_with_logits(n, target_idxs)
    dw = m.T.dot(dlog)
    
    return dw
# %%
def optimize(theta, grad, lr=0.03):
    theta -= grad * lr
    return theta
# %%
embeddings = np.random.sample((vocab_size, embed_dim))
#%%
embeddings.shape
# %%
embeddings
# %%
theta = np.random.uniform(-1, 1, (embed_dim*2, vocab_size))
# %%
theta.shape
# %%
epoch_losses = {}
epochs = 80
for epoch in range(epochs):
    losses = []
    for context, target in bag:
        context_idxs = np.array([tokenizer.word_index[word] for word in context])
        preds = forward(context_idxs, theta, embeddings)

        target_idxs = np.array([tokenizer.word_index[target]])
        loss = NLLLoss(preds[-1], target_idxs)

        losses.append(loss)

        grad = backward(preds, theta, target_idxs)
        theta = optimize(theta, grad, lr=0.03)

    epoch_losses[epoch] = losses
# %%
context_idxs
# %%

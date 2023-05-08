import tensorflow as tf
import numpy as np
tf.random.set_seed(1337)

#hyperparameters
block_size = 256
batch_size = 32
n_embed = 16
n_units = 2048
dropout = 0.1
new_tokens = 100

#using tiny shakespeare as input data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#set vocab_size
chars = sorted(list(set(text)))
vocab_size = len(chars)

#tokenization
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

#split into x and y
def split_sequence(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

#make dataset
def make_ds(text, block_size, batch_size):
    n = int(0.9*len(text))
    train_data = encode(text[:n])
    val_data = encode(text[n:])
    train_ds = tf.data.Dataset.from_tensor_slices(train_data)
    val_ds = tf.data.Dataset.from_tensor_slices(val_data)
    sequences_train = train_ds.batch(block_size+1, drop_remainder=True)
    sequences_val = val_ds.batch(block_size+1, drop_remainder=True)
    train_ds_final = sequences_train.map(split_sequence)
    val_ds_final = sequences_val.map(split_sequence)
    return train_ds_final.batch(batch_size), val_ds_final.batch(batch_size)

#generate from the model
def generate(model, block_size, context, new_tokens):
    context = tf.expand_dims(context[0,-block_size:],0)
    for i in range(new_tokens):
        logits = model.predict(tf.expand_dims(context[0,-block_size:],0),verbose=0)
        logits = logits[0,-1,:]
        pred = tf.random.categorical(tf.math.log(tf.expand_dims(logits,0)),1)
        pred = tf.cast(pred,dtype=tf.int32)
        context = tf.concat([context,pred],axis=1)
    return context

#make dataset
train_ds, val_ds = make_ds(text, block_size, batch_size)

#Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, n_embed, input_length=block_size),
    tf.keras.layers.LSTM(n_units, return_sequences=True),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Dense(vocab_size),
    tf.keras.layers.Softmax(),
    tf.keras.layers.Dropout(dropout)
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.SparseCategoricalCrossentropy())

model.fit(train_ds, epochs=20, batch_size=batch_size, validation_data=val_ds)

#small example
for context, _ in val_ds.take(1):    
    print(decode(context[0].numpy()))
    result = generate(model, block_size , context, 5 * new_tokens)
    print("-" * 20)
    print(decode(result.numpy()[0]))

#save bigger example to file
result = generate(model, block_size , context, 10000)
open('more.txt', 'w').write(decode(result.numpy()[0]))
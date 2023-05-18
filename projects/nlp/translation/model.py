from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
#set seeds
tf.random.set_seed(1337)
np.random.seed(1337)

#hyperparameters
batch_size = 64
vocab_size = 8192
n_units = 1024
n_embed = 64
n_heads = 4
epochs = 30
steps = 100 #steps per epooh

#load data
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'
text = path_to_file.read_text(encoding="utf-8")

lines = text.splitlines()
pairs = [line.split('\t') for line in lines]

#shuffle data
pairs = np.array(pairs)
np.random.shuffle(pairs)

#split data into spa and eng texts
target = pairs[:,0]
context = pairs[:,1]

#split into train and val, covert to tfds
n = int(0.8*len(context))

train_raw = (
    tf.data.Dataset
    .from_tensor_slices((context[:n], target[:n]))
    .shuffle(len(context))
    .batch(batch_size))
val_raw = (
    tf.data.Dataset
    .from_tensor_slices((context[n:], target[n:]))
    .shuffle(len(context))
    .batch(batch_size))

#preprocess, add BOS and EOS
def preprocess(text):
    # Split accented characters
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

#vectorize
processor_spa = tf.keras.layers.TextVectorization(
    standardize=preprocess,
    max_tokens=vocab_size,
    ragged=True)

processor_spa.adapt(train_raw.map(lambda context, target: context))

processor_eng = tf.keras.layers.TextVectorization(
    standardize=preprocess,
    max_tokens=vocab_size,
    ragged=True)

processor_eng.adapt(train_raw.map(lambda context, target: target))

#move eng sequences 1 token to right for training next word prediction based on context
def prepare_text(context, target):
    context = processor_spa(context).to_tensor()
    target = processor_eng(target)
    targ_in = target[:,:-1].to_tensor()
    targ_out = target[:,1:].to_tensor()
    return (context, targ_in), targ_out

train_ds = train_raw.map(prepare_text)
val_ds = val_raw.map(prepare_text)

#word to id and id to word dicts for both languages
word_to_id_eng = tf.keras.layers.StringLookup(
        vocabulary=processor_eng.get_vocabulary(),
        mask_token='', oov_token='[UNK]')
word_to_id_spa = tf.keras.layers.StringLookup(
        vocabulary=processor_spa.get_vocabulary(),
        mask_token='', oov_token='[UNK]')
id_to_word_eng = tf.keras.layers.StringLookup(
        vocabulary=processor_eng.get_vocabulary(),
        mask_token='', oov_token='[UNK]',
        invert=True)
id_to_word_spa = tf.keras.layers.StringLookup(
        vocabulary=processor_spa.get_vocabulary(),
        mask_token='', oov_token='[UNK]',
        invert=True)

#MODEL
#Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, units, n_embed):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.units = units
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, n_embed, mask_zero=True)
        self.rnn = tf.keras.layers.Bidirectional(merge_mode='sum',
            layer=tf.keras.layers.GRU(units, return_sequences=True))      

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)
        return x
    
#CrossAttention
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=n_heads, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        attn_output = self.mha(
            query=x,
            value=context,
            return_attention_scores=False)

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

#Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, units, n_embed):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.units = units
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, n_embed, mask_zero=True)
        self.rnn = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.attention = CrossAttention(units)
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], word_to_id_spa('[START]'))
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embed = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(embed)[0]
    
    def get_next_token(self, context, next_token, done, state):
        logits, state = self(context, next_token, state = state, return_state=True) 
        logits = logits[:, -1, :]
        next_token = tf.random.categorical(logits, num_samples=1)
        done = done | (next_token == word_to_id_eng('[END]'))
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
        return next_token, done, state

    def call(self, context, x, state=None, return_state=False):  
        x = self.embedding(x)
        x, state = self.rnn(x, initial_state=state)
        x = self.attention(x, context)
        logits = self.output_layer(x)

        if return_state:
            return logits, state
        else:
            return logits

#Model
class Model(tf.keras.Model):
    def __init__(self, units, n_embed):
        super().__init__()
        self.encoder = Encoder(units, n_embed)
        self.decoder = Decoder(units, n_embed)

    def get_initial_state(self, context):
        return self.decoder.get_initial_state(context)
    
    def get_next_token(self, context, next_token, done, state):
        return self.decoder.get_next_token(context, next_token, done, state)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)
        return logits
    
#Training
model = Model(n_units, n_embed)
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=["accuracy"])
model.fit(train_ds.repeat(), epochs=epochs, steps_per_epoch=steps, validation_data=val_ds, validation_steps=20)

#test the model
for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
    pass

#pass spa sequecne through encoder
ex_context = model.encoder(ex_context_tok)

#generate translations
next_token, done, state = model.get_initial_state(ex_context)
tokens = []

for n in range(10):
    next_token, done, state = model.get_next_token(
        ex_context, next_token, done, state)
    tokens.append(next_token)

tokens = tf.concat(tokens, axis=-1)

#transform tokens into words
def tokens_to_text(tokens, language):
    words = id_to_word_eng(tokens) if language=="eng" else id_to_word_spa(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
    return result.numpy()

print(tokens_to_text(tokens, language="eng"))
print(tokens_to_text(ex_context_tok,language="spa"))
import numpy as np 
import pandas as pd 
import tensorflow as tf
import re
import string
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
#seeds
tf.random.set_seed(1337)
np.random.seed(1337)

#hyperparameters
n_embed = 32
batch_size = 32
max_len = 32

#load data
data = pd.read_csv('data/Corona_NLP_train.csv', encoding='latin-1')
data_test = pd.read_csv('data/Corona_NLP_test.csv', encoding='latin-1')

#split data
data_val = data[int(0.8*len(data)):]
data_train = data[:int(0.8*len(data))]

#Preprocessing
stopwords = stopwords.words('english')

def preprocess(text):
    text = text.lower() #converting input to lowercase
    text = re.sub(r'\([^)]*\)', '', text) #Removing punctuations and special characters.
    text = re.sub('"','', text) #Removing double quotes.
    text = re.sub(r"'s\b","",text) #Eliminating apostrophe.
    text = re.sub(r"@[A-Za-z0-9_]+", "", text) #Removing twitter handles
    text = re.sub("[^a-zA-Z]", " ", text) #Removing non-alphabetical characters
    text = ' '.join([word for word in text.split() if word not in stopwords]) #Removing stopwords.
    return text


clean_text_train = [preprocess(text) for text in data_train["OriginalTweet"].values]
clean_text_val = [preprocess(text) for text in data_val["OriginalTweet"].values]
clean_text_test = [preprocess(text) for text in data_test["OriginalTweet"].values]

#label encoding
Sentiment = np.unique(data.Sentiment.values)
stoi = {i:s for s,i in enumerate(Sentiment)}
itos = { i:ch for i,ch in enumerate(Sentiment)}

encode = lambda s: [stoi[i] for i in s]
decode = lambda l: itos[l]

labels_train = encode(data_train.Sentiment.values)
labels_val = encode(data_val.Sentiment.values)
labels_test = encode(data_test.Sentiment.values)

#Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_text_train)
x_train_seq = tokenizer.texts_to_sequences(clean_text_train)
x_val_seq = tokenizer.texts_to_sequences(clean_text_val)
x_test_seq = tokenizer.texts_to_sequences(clean_text_test)

#Padding to max_len
vocab_size = len(tokenizer.word_index) + 1
x_train_pad = pad_sequences(x_train_seq, maxlen=max_len, padding='post')
x_val_pad = pad_sequences(x_val_seq, maxlen=max_len, padding='post')
x_test_pad = pad_sequences(x_test_seq, maxlen=max_len, padding='post')

labels_train = np.array(labels_train)
labels_val = np.array(labels_val)
labels_test = np.array(labels_test)

labels_train = np.reshape(labels_train, (len(labels_train),1))
labels_val = np.reshape(labels_val, (len(labels_val),1))
labels_test = np.reshape(labels_test, (len(labels_test),1))


#Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, n_embed,input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(5),
    tf.keras.layers.Softmax(),
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x_train_pad, labels_train, epochs=7, batch_size=batch_size, validation_data=(x_val_pad, labels_val))

#get final accuracy
loss, acc = model.evaluate(x_test_pad, labels_test)
print(f"Accuracy on validation set: {acc}")

#write examples
choices = np.random.choice(data_test["OriginalTweet"].values,10)
choices_vec = tokenizer.texts_to_sequences(choices)
choices_vec = pad_sequences(choices_vec, maxlen=max_len, padding='post')
preds = model.predict(choices_vec, verbose=0)
for choice,pred in zip(choices,preds):
    print(choice)
    print(f"Sentiment: {decode(np.argmax(pred))}")
    print("-" * 200)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_datasets as tfds
#set seed
tf.random.set_seed(1337)

#hyperparameters
IMG_SIZE = 256
batch_size = 32
dropout = 0.2

#get data
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str
num_classes = metadata.features['label'].num_classes

#resize into IMG_SIZE x IMG_SIZE and rescale to [0...1]
preproc = tf.keras.Sequential([
  tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
  tf.keras.layers.Rescaling(1/255),
])

def preproc_dataset(dataset):
    dataset = dataset.map(lambda x, y: (preproc(x), y))
    dataset = dataset.batch(batch_size)
    return dataset

#apply
train_ds_preproc = preproc_dataset(train_ds).shuffle(1024)
val_ds_preproc = preproc_dataset(val_ds).shuffle(1024)
test_ds_preproc = preproc_dataset(test_ds).shuffle(1024)

#rotating and flipping to get more training examples
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.1),
])

#apply several times
combined_dataset = train_ds_preproc
for _ in range(5):
    ds_da = train_ds_preproc.map(lambda x, y: (data_augmentation(x, training=True), y))
    combined_dataset = combined_dataset.concatenate(ds_da).shuffle(1024)

#Model
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',input_shape=(IMG_SIZE, IMG_SIZE, 3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes),
  tf.keras.layers.Dropout(dropout)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

model.fit(combined_dataset, epochs=8, batch_size=batch_size,validation_data=val_ds_preproc)

#get final accuracy
loss, acc = model.evaluate(test_ds_preproc)
print(f"Accuracy on test set: {acc}")

#plot examples
fig, axes = plt.subplots(5, 5, figsize=(12,12))
for ax, (image, label) in zip(axes.flat,test_ds.take(25)):
    ax.imshow(image)
    image_preproc = tf.expand_dims(preproc(image),0)
    pred = model.predict(image_preproc, verbose=0)
    pred = np.argmax(pred)
    ax.set_title(get_label_name(pred))
    ax.axis('off')
plt.show()
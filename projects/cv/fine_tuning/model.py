import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
#set seeds
tf.random.set_seed(1337)
np.random.seed(1337)

#hyperparameters
IMG_SIZE = 160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
batch_size = 32
learning_rate = 1e-4
epochs_start = 10
epochs_end = 10
unfreeze_at = 120
dropout = 0.2
n_units = 128

#load data
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features['label'].num_classes
get_label_name = metadata.features['label'].int2str

#preproc data - resize and rescale to -1...1
preproc = tf.keras.Sequential([
  tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
  tf.keras.layers.Rescaling(scale=1./127.5, offset=-1),
])


AUTOTUNE = tf.data.AUTOTUNE
def preproc_dataset(dataset):
    dataset = dataset.map(lambda x, y: (preproc(x), y)).shuffle(1024).batch(batch_size)
    return dataset


train_ds_preproc = preproc_dataset(train_ds)
val_ds_preproc = preproc_dataset(val_ds)
test_ds_preproc = preproc_dataset(test_ds)

#data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])

#Model
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
#freeze base model
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(num_classes, activation="softmax")
mid_layer = tf.keras.layers.Dense(n_units, activation="relu")

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(dropout)(x)
x = mid_layer(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])


history = model.fit(train_ds_preproc,epochs=epochs_start,validation_data=val_ds_preproc)

#unfreeze last n layers
base_model.trainable = True
for layer in base_model.layers[:unfreeze_at]:
    layer.trainable = False

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate/10),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
history2 = model.fit(train_ds_preproc,epochs=epochs_start+epochs_end,validation_data=val_ds_preproc,initial_epoch=history.epoch[-1])

#test the model
loss, acc = model.evaluate(test_ds_preproc)
print(f"Accuracy on test set: {acc}")

#print examples
fig, axes = plt.subplots(5, 5, figsize=(12,12))
for ax, (image, label) in zip(axes.flat,test_ds.take(25)):
    ax.imshow(image)
    image_preproc = tf.expand_dims(preproc(image),0)
    pred = model.predict(image_preproc, verbose=0)
    pred = np.argmax(pred)
    ax.set_title(get_label_name(pred))
    ax.axis('off')
plt.show()


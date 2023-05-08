import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
#seeds
tf.random.set_seed(1337)
random.seed(1337)

#hyperparameters
epochs = 10
batch_size = 32
dropout = 0.2

#get data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
num_classes = 10

#rescale images
train_images = train_images / 255
test_images = test_images / 255

#Model
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28,28,1),input_shape=(28, 28)),
    tf.keras.layers.Conv2D(16,2,padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32,2,padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D(),    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes),
    tf.keras.layers.Softmax(),
    tf.keras.layers.Dropout(dropout),
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,epochs=epochs,batch_size=batch_size,validation_data=(test_images,test_labels))

#get final accuracy
loss, acc = model.evaluate(test_images, test_labels)
print(f"Accuracy on validation set: {acc}")

#plot examples
fig, axes = plt.subplots(5, 5, figsize=(12,12))
for ax in axes.flat:
    i = random.randint(0, len(test_labels))
    image = test_images[i]
    image_batch = tf.expand_dims(image, 0)
    pred = model.predict(image_batch,verbose=0)
    ax.imshow(image, cmap='gray')
    ax.set_title(np.argmax(pred))
    ax.axis('off')
plt.show()
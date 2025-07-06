import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from utils import (
    get_dataset,
    preprocess_input,
    get_callbacks,
    eval_and_save,
    data_augmentation,
    mixup,
    preprocess_label
)

cinic_directory = "data/CINIC10"
IMG_HEIGHT = 32
IMG_WIDTH = 32
SEED = 1337


def main():
    cinic_train_raw = get_dataset(cinic_directory + "/train")
    cinic_val_raw = get_dataset(cinic_directory + "/valid")
    cinic_test_raw = get_dataset(cinic_directory + "/test")
    type = "CNN"

    # Model 1
    path = "cnn1.keras"
    config = {
        "Data Augmentation": "No",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 2
    path = "cnn2.keras"
    config = {
        "Data Augmentation": "No",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 32,
    }

    cinic_train = (
        cinic_train_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 3
    path = "cnn3.keras"
    config = {
        "Data Augmentation": "No",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 4
    path = "cnn4.keras"
    config = {
        "Data Augmentation": "No",
        "Regularization": "L2",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                256,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 5
    path = "cnn5.keras"
    config = {
        "Data Augmentation": "No",
        "Regularization": "L2 + Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                256,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 6
    path = "cnn6.keras"
    config = {
        "Data Augmentation": "No",
        "Regularization": "Gradient Clipping + Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["Learning Rate"], clipvalue=1
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 7
    path = "cnn7.keras"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }
    
    cinic_train = (
        cinic_train_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
        .map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 8
    path = "cnn8.keras"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 32,
    }

    cinic_train = (
        cinic_train_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
        .map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 9
    path = "cnn9.keras"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
        .map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 10
    path = "cnn10.keras"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "L2",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
        .map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                256,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 11
    path = "cnn11.keras"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "L2 + Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
        .map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                256,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 12
    path = "cnn12.keras"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "Gradient Clipping + Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
        .map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.map(preprocess_input)
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["Learning Rate"], clipvalue=1
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 13
    path = "cnn13.keras"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(mixup)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 14
    path = "cnn14.keras"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 32,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(mixup)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 15
    path = "cnn15.keras"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(mixup)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 16
    path = "cnn16.keras"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "L2",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(mixup)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                256,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 17
    path = "cnn17.keras"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "L2 + Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(mixup)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                256,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)

    # Model 18
    path = "cnn18.keras"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "Gradient Clipping + Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(mixup)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .map(preprocess_input)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["Learning Rate"], clipvalue=1
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=100,
        callbacks=get_callbacks("models/" + path),
    )

    eval_and_save(type, cinic_test, config, history, path)


if __name__ == "__main__":
    for i in range(3):
        main()

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from utils import (
    get_dataset,
    get_callbacks_weights,
    eval_and_save_weights,
    data_augmentation,
    mixup,
    preprocess_label,
    unfreeze_model,
)

cinic_directory = "data/CINIC10"
IMG_HEIGHT = 32
IMG_WIDTH = 32
SEED = 1337


def main():
    cinic_train_raw = get_dataset(cinic_directory + "/train")
    cinic_val_raw = get_dataset(cinic_directory + "/valid")
    cinic_test_raw = get_dataset(cinic_directory + "/test")
    type = "MobileNet"

    # Model 1
    path = "mobilenet1_pretrained.weights.h5"
    config = {
        "Data Augmentation": "No",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", dropout_rate=0
    )
    base_model.trainable = False
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Resizing(224, 224),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
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
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)
    unfreeze_model(model)

    path = "mobilenet1_finetuned.weights.h5"
    config = {
        "Data Augmentation": "No",
        "Regularization": "No",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=config["Learning Rate"] / 10
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)

    # Model 2
    path = "mobilenet2_pretrained.weights.h5"
    config = {
        "Data Augmentation": "No",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.00005,
        "Batch Size": 32,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", dropout_rate=0
    )
    base_model.trainable = False
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Resizing(224, 224),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
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
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)
    unfreeze_model(model)

    path = "mobilenet2_finetuned.weights.h5"
    config = {
        "Data Augmentation": "No",
        "Regularization": "No",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.00005,
        "Batch Size": 32,
    }
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=config["Learning Rate"] / 10
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)

    # Model 3
    path = "mobilenet3_pretrained.weights.h5"
    config = {
        "Data Augmentation": "No",
        "Regularization": "Gradient Clipping + Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Resizing(224, 224),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
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
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)
    unfreeze_model(model)

    path = "mobilenet3_finetuned.weights.h5"
    config = {
        "Data Augmentation": "No",
        "Regularization": "Gradient Clipping + Dropout",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=config["Learning Rate"] / 10, clipvalue=1
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)

    # Model 4
    path = "mobilenet4_pretrained.weights.h5"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", dropout_rate=0
    )
    base_model.trainable = False
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Resizing(224, 224),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
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
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)
    unfreeze_model(model)

    path = "mobilenet4_finetuned.weights.h5"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "No",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=config["Learning Rate"] / 10
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)

    # Model 5
    path = "mobilenet5_pretrained.weights.h5"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.00005,
        "Batch Size": 32,
    }

    cinic_train = (
        cinic_train_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", dropout_rate=0
    )
    base_model.trainable = False
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Resizing(224, 224),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
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
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)
    unfreeze_model(model)

    path = "mobilenet5_finetuned.weights.h5"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "No",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.00005,
        "Batch Size": 32,
    }
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=config["Learning Rate"] / 10
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)

    # Model 6
    path = "mobilenet6_pretrained.weights.h5"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "'Gradient Clipping + Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
        .batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Resizing(224, 224),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
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
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)
    unfreeze_model(model)

    path = "mobilenet6_finetuned.weights.h5"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "'Gradient Clipping + Dropout",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=config["Learning Rate"] / 10, clipvalue=1
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)

    # Model 7
    path = "mobilenet7_pretrained.weights.h5"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(mixup)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", dropout_rate=0
    )
    base_model.trainable = False
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Resizing(224, 224),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
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
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)
    unfreeze_model(model)

    path = "mobilenet7_finetuned.weights.h5"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "No",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=config["Learning Rate"] / 10
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)

    # Model 8
    path = "mobilenet8_pretrained.weights.h5"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "No",
        "Optimizer": "Adam",
        "Learning Rate": 0.00005,
        "Batch Size": 32,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(mixup)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", dropout_rate=0
    )
    base_model.trainable = False
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Resizing(224, 224),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
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
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)
    unfreeze_model(model)

    path = "mobilenet8_finetuned.weights.h5"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "No",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.00005,
        "Batch Size": 32,
    }
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=config["Learning Rate"] / 10
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)

    # Model 9
    path = "mobilenet9_pretrained.weights.h5"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "Gradient Clipping + Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(mixup)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(preprocess_label)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Resizing(224, 224),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
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
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)
    unfreeze_model(model)

    path = "mobilenet9_finetuned.weights.h5"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "Gradient Clipping + Dropout",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=config["Learning Rate"] / 10, clipvalue=1
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        cinic_train,
        validation_data=cinic_val,
        epochs=30,
        callbacks=get_callbacks_weights("models/" + path),
    )

    model = eval_and_save_weights(type, model, cinic_test, config, history, path)


if __name__ == "__main__":
    main()

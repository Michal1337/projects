import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from utils import (
    get_dataset,
    get_callbacks,
    eval_and_save_vit,
    data_augmentation,
    mixup,
    preprocess_label,
    unfreeze_model_vit,
)
from vit_keras import vit

cinic_directory = "data/CINIC10"
IMG_HEIGHT = 32
IMG_WIDTH = 32
SEED = 1337


def main():
    cinic_train_raw = get_dataset(cinic_directory + "/train")
    cinic_val_raw = get_dataset(cinic_directory + "/valid")
    cinic_test_raw = get_dataset(cinic_directory + "/test")
    type = "ViT"

    # Model 1
    path = "ViT1_pretrained.keras"
    config = {
        "Data Augmentation": "No",
        "Regularization": "Dropout + Gradient Clipping",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(lambda x, y: ((vit.preprocess_inputs(x)), y))
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(lambda x, y: ((vit.preprocess_inputs(x)), y))
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(lambda x, y: ((vit.preprocess_inputs(x)), y))
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = vit.vit_b32(
        image_size=224,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    base_model.trainable = False
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(224, 224),
            base_model,
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
        epochs=20,
        callbacks=get_callbacks("models/" + path),
    )

    model = eval_and_save_vit(type, cinic_test, config, path)
    model = unfreeze_model_vit(model)

    path = "ViT1_finetuned.keras"
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
        epochs=20,
        callbacks=get_callbacks("models/" + path),
    )

    model = eval_and_save_vit(type, cinic_test, config, history, path)

    # Model 2
    path = "ViT2_pretrained.keras"
    config = {
        "Data Augmentation": "Type 1",
        "Regularization": "Dropout",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
        .batch(config["Batch Size"])
        .map(lambda x, y: ((vit.preprocess_inputs(x)), y))
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(lambda x, y: ((vit.preprocess_inputs(x)), y))
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(lambda x, y: ((vit.preprocess_inputs(x)), y))
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = vit.vit_b32(
        image_size=224,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    base_model.trainable = False
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(224, 224),
            base_model,
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
        epochs=20,
        callbacks=get_callbacks("models/" + path),
    )

    model = eval_and_save_vit(type, cinic_test, config, history, path)
    model = unfreeze_model_vit(model)

    path = "ViT2_finetuned.keras"
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
        epochs=20,
        callbacks=get_callbacks("models/" + path),
    )

    model = eval_and_save_vit(type, cinic_test, config, history, path)

    # Model 3
    path = "ViT3_pretrained.keras"
    config = {
        "Data Augmentation": "Type 2",
        "Regularization": "Dropout",
        "Optimizer": "Adam + RMSprop",
        "Learning Rate": 0.0001,
        "Batch Size": 256,
    }

    cinic_train = (
        cinic_train_raw.batch(config["Batch Size"])
        .map(mixup)
        .map(lambda x, y: ((vit.preprocess_inputs(x)), y))
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_val = (
        cinic_val_raw.batch(config["Batch Size"])
        .map(lambda x, y: ((vit.preprocess_inputs(x)), y))
        .map(preprocess_label)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    cinic_test = (
        cinic_test_raw.batch(config["Batch Size"])
        .map(lambda x, y: ((vit.preprocess_inputs(x)), y))
        .map(preprocess_label)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    base_model = vit.vit_b32(
        image_size=224,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    base_model.trainable = False
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(224, 224),
            base_model,
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
        epochs=20,
        callbacks=get_callbacks("models/" + path),
    )

    model = eval_and_save_vit(type, cinic_test, config, history, path)
    model = unfreeze_model_vit(model)

    path = "ViT3_finetuned.keras"
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
        epochs=20,
        callbacks=get_callbacks("models/" + path),
    )

    model = eval_and_save_vit(type, cinic_test, config, history, path)


if __name__ == "__main__":
    main()

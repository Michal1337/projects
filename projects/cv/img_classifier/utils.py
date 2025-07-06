import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
from typing import Tuple, Dict, Union, List

SEED = 1337
IMG_HEIGHT = 32
IMG_WIDTH = 32

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomBrightness(factor=0.2),
    ]
)


def get_dataset(path: str) -> tf.data.Dataset:
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        batch_size=None,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        label_mode="int",
        seed=SEED,
    )
    return dataset


def preprocess_input(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    image = image / 255
    image = (image - cinic_mean) / cinic_std
    return image, label


def get_callbacks(
    path: str,
) -> list[tf.keras.callbacks.Callback, tf.keras.callbacks.Callback]:
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        path, save_best_only=True, monitor="val_accuracy", mode="max"
    )
    return [early_stopping, checkpoint]


def get_callbacks_weights(
    path: str,
) -> list[tf.keras.callbacks.Callback, tf.keras.callbacks.Callback]:
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        path,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
    )
    return [early_stopping, checkpoint]


def eval_and_save(
    type: str,
    cinic_test: tf.data.Dataset,
    config: Dict[str, Union[int, str, float]],
    history: Dict[str, List[float]],
    path: str,
):
    model = tf.keras.models.load_model("models/" + path)
    loss, acc = model.evaluate(cinic_test)

    history = pd.DataFrame(history.history)
    history.to_csv(f"history/{path.split('.')[0]}.csv")

    with open("results.csv", "a") as f:
        f.write(f"{type};{model.count_params()};{loss};{acc};{config};{path}\n")


def eval_and_save_weights(
    type: str,
    model: tf.keras.Model,
    cinic_test: tf.data.Dataset,
    config: Dict[str, Union[int, str, float]],
    history: Dict[str, List[float]],
    path: str,
) -> tf.keras.Model:
    model.load_weights("models/" + path)
    loss, acc = model.evaluate(cinic_test)

    history = pd.DataFrame(history.history)
    history.to_csv(f"history/{path.split('.')[0]}.csv")

    with open("results.csv", "a") as f:
        f.write(f"{type};{model.count_params()};{loss};{acc};{config};{path}\n")

    return model


def eval_and_save_vit(
    type: str,
    cinic_test: tf.data.Dataset,
    config: Dict[str, Union[int, str, float]],
    path: str,
) -> tf.keras.Model:
    model = tf.keras.models.load_model("models/" + path, safe_mode=False)
    loss, acc = model.evaluate(cinic_test)

    # history = pd.DataFrame(history.history)
    # history.to_csv(f"history/{path.split('.')[0]}.csv")

    with open("results.csv", "a") as f:
        f.write(f"{type};{model.count_params()};{loss};{acc};{config};{path}\n")

    return model


def mixup(
    batch_images: tf.Tensor, batch_labels: tf.Tensor, alpha: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(batch_images)[0]
    lam = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1, 1, 1])

    batch_labels = tf.one_hot(batch_labels, 10)
    shuffled_indices = tf.random.shuffle(tf.range(batch_size))

    mixed_images = lam * batch_images + (1.0 - lam) * tf.gather(
        batch_images, shuffled_indices
    )

    lam = tf.reshape(lam, (batch_size, 1))

    mixed_labels = lam * batch_labels + (1.0 - lam) * tf.gather(
        batch_labels, shuffled_indices
    )

    return mixed_images, mixed_labels


def preprocess_label(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return image, tf.one_hot(label, 10)


def unfreeze_model(model: tf.keras.Model) -> tf.keras.Model:
    model.get_layer("MobilenetV3small").trainable = True
    for layer in model.get_layer("MobilenetV3small").layers[:100]:
        layer.trainable = False
    return model


def unfreeze_model_vit(model: tf.keras.Model) -> tf.keras.Model:
    model.get_layer("vit-b32").trainable = True
    for layer in model.get_layer("vit-b32").layers[:14]:
        layer.trainable = False
    return model

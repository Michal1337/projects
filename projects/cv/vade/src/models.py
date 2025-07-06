import tensorflow as tf
import numpy as np
import math


class Sampling(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super(Sampling, self).__init__()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, latent_dim: int, autoencoder: bool) -> None:
        super(Encoder, self).__init__()
        enc_layers = [
            tf.keras.layers.InputLayer(shape=(input_dim,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(2048, activation="relu"),
        ]
        if autoencoder:
            enc_layers.append(tf.keras.layers.Dense(latent_dim))

        self.encoder = tf.keras.Sequential([*enc_layers])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.encoder(x)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_dim: int, latent_dim: int) -> None:
        super(Decoder, self).__init__()
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(latent_dim,)),
                tf.keras.layers.Dense(2048, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(output_dim, activation="sigmoid"),
            ]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.decoder(x)


class Autoencoder(tf.keras.Model):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, True)
        self.decoder = Decoder(input_dim, latent_dim)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


class VAE(tf.keras.Model):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, False)
        self.decoder = Decoder(input_dim, latent_dim)
        self.sampling = Sampling()
        self.z_mean = tf.keras.layers.Dense(latent_dim)
        self.z_log_var = tf.keras.layers.Dense(latent_dim)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.encoder(inputs)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var, z


class VaDE(tf.keras.Model):
    def __init__(self, input_dim: int, latent_dim: int, n_clusters: int) -> None:
        super(VaDE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, False)
        self.decoder = Decoder(input_dim, latent_dim)
        self.sampling = Sampling()
        self.z_mean = tf.keras.layers.Dense(latent_dim)
        self.z_log_var = tf.keras.layers.Dense(latent_dim)
        self.latent_dim = latent_dim

        # Cluster parameters
        self.pi_ = self.add_weight(
            shape=(n_clusters,), initializer="ones", trainable=True
        )
        self.mu_c = self.add_weight(
            shape=(n_clusters, latent_dim), initializer="random_normal", trainable=True
        )
        self.log_var_c = self.add_weight(
            shape=(n_clusters, latent_dim), initializer="random_normal", trainable=True
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.encoder(inputs)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var, z


    def classify(self, x: np.ndarray, n_samples: int = 10) -> tf.Tensor:
        x = self.encoder(x)
        mu, logvar = self.z_mean(x), self.z_log_var(x)
        
        z_samples = [self.sampling([mu, logvar]) for _ in range(n_samples)]
        z = tf.stack(z_samples, axis=1)

        log_p_z_given_c = (
            tf.reduce_sum(-0.5 * self.log_var_c, axis=1) - 
            0.5 * (math.log(2 * math.pi) + 
                   tf.reduce_sum(
                       tf.pow((tf.expand_dims(z, 2) - self.mu_c), 2) / 
                       (tf.exp(self.log_var_c) + 1e-9), axis=3))
        )
        weights = tf.nn.softmax(self.pi_, axis=0)
        p_z_c = tf.exp(log_p_z_given_c) * weights
        y = p_z_c / (tf.reduce_sum(p_z_c, axis=2, keepdims=True) + 1e-9)
        y = tf.reduce_sum(y, axis=1)
        pred = tf.argmax(y, axis=1)

        return pred
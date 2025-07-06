import math
import tensorflow as tf
from sklearn.mixture import GaussianMixture


from models import Autoencoder, VaDE
from utils import *


@tf.function
def elbo_vade(x, recon_x, mu, logvar, z, mu_c, log_var_c, pi_):
    weights = tf.nn.softmax(pi_, axis=0)
    z = tf.expand_dims(z, 1)
    batch_size = tf.shape(x)[0]
    h = z - mu_c
    h = tf.exp(-0.5 * tf.reduce_sum((h * h / tf.exp(log_var_c)), axis=2))
    h = h / tf.exp(tf.reduce_sum(0.5 * log_var_c, axis=1))
    p_z_given_c = h / (2 * math.pi)
    p_z_c = p_z_given_c * weights + 1e-9
    gamma = p_z_c / (tf.reduce_sum(p_z_c, axis=1, keepdims=True))
    h = tf.expand_dims(tf.exp(logvar), 1) + tf.square(
        tf.expand_dims(mu, 1) - mu_c
    )
    h = tf.reduce_sum(log_var_c + h / tf.exp(log_var_c), axis=2)
    kl_div = (
        0.5 * tf.reduce_sum(gamma * h)
        - tf.reduce_sum(gamma * tf.math.log(weights + 1e-9))
        + tf.reduce_sum(gamma * tf.math.log(gamma + 1e-9))
        - 0.5 * tf.reduce_sum(1 + logvar)
    )

    recon_loss = tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(x, recon_x) * x.shape[1]
    )

    loss = recon_loss + kl_div
    loss = loss / tf.cast(batch_size, tf.float32)
    return loss


def pretrain(vade, autoencoder, x_train, n_clusters):
    # encoder
    for i in range(6):
        vade.layers[0].weights[i].assign(autoencoder.layers[0].weights[i])

    # decoder
    for i in range(8):
        vade.layers[1].weights[i].assign(autoencoder.layers[1].weights[i])

    latent_representations = autoencoder.encoder(x_train).numpy()
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag")
    gmm.fit(latent_representations)
    vade.mu_c.assign(gmm.means_)
    vade.log_var_c.assign(np.log(gmm.covariances_))
    vade.pi_.assign(gmm.weights_)


def main(
    num_epochs_auto, num_epochs_vade, batch_size, learning_rate_auto, learning_rate_vade
):
    original_dim = 784
    latent_dim = 10
    n_clusters = 10
    x_train, y_train, x_test, y_test = get_data()
    autoencoder = Autoencoder(original_dim, latent_dim)
    vade = VaDE(original_dim, latent_dim, n_clusters)

    # Train autoencoder
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate_auto), loss="mse"
    )
    autoencoder.fit(
        x_train,
        x_train,
        epochs=num_epochs_auto,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=get_callbacks(5, "autoencoder.weights.h5"),
    )

    # Pretrain VaDE
    autoencoder.load_weights("../models/autoencoder.weights.h5")
    pretrain(vade, autoencoder, x_train, n_clusters)
    print("Pretraining VaDE done")

    # Train VaDE
    optimizer = tf.keras.optimizers.Adam(learning_rate_vade)
    best_loss = np.inf
    for epoch in range(num_epochs_vade):
        total_loss = 0
        for start in range(0, len(x_train), batch_size):
            end = start + batch_size
            x_batch = x_train[start:end]
            with tf.GradientTape() as tape:
                x_reconstructed, mu, logvar, z = vade(x_batch)
                loss = elbo_vade(x_batch, x_reconstructed, mu, logvar, z, vade.mu_c, vade.log_var_c, vade.pi_)
            total_loss += loss.numpy() * len(x_batch)
            gradients = tape.gradient(loss, vade.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vade.trainable_variables))
        if total_loss < best_loss:
            best_loss = total_loss
            vade.save_weights(f"../models/vade.weights.h5")
        print(
            f"Epoch [{epoch+1}/{num_epochs_vade}], Loss: {total_loss / len(x_train):.4f}"
        )


if __name__ == "__main__":
    batch_size = 128
    num_epochs_auto = 100
    num_epochs_vade = 100
    learning_rate_auto = 1e-3
    learning_rate_vade = 1e-4
    main(
        num_epochs_auto,
        num_epochs_vade,
        batch_size,
        learning_rate_auto,
        learning_rate_vade,
    )

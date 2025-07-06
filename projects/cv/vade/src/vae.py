import tensorflow as tf

from models import VAE
from utils import *


@tf.function
def elbo_vae(x, x_decoded, z_mean, z_log_var):
    loss_term = tf.reduce_sum(
        x * tf.math.log(x_decoded + 1e-19)
        + (1 - x) * tf.math.log(1 - x_decoded + 1e-19),
        1,
    )
    KL_term = 0.5 * tf.reduce_sum(
        -z_log_var + tf.exp(z_log_var) + tf.square(z_mean) - 1, 1
    )
    return tf.reduce_mean(loss_term - KL_term)


def main(num_epochs, batch_size, learning_rate):
    original_dim = 784
    latent_dim = 10
    x_train, y_train, x_test, y_test = get_data()
    vae = VAE(original_dim, latent_dim)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    best_loss = np.inf
    for epoch in range(num_epochs):
        total_loss = 0
        for start in range(0, len(x_train), batch_size):
            end = start + batch_size
            x_batch = x_train[start:end]
            with tf.GradientTape() as tape:
                x_reconstructed, mu, logvar, z = vae(x_batch)
                loss = -elbo_vae(x_batch, x_reconstructed, mu, logvar)
            total_loss += loss.numpy() * len(x_batch)
            gradients = tape.gradient(loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        if total_loss < best_loss:
            best_loss = total_loss
            vae.save_weights(f'../models/vae.weights.h5')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(x_train):.4f}')


if __name__ == '__main__':
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3
    main(num_epochs, batch_size, learning_rate)


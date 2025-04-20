import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from tensorflow.python.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  
y = iris.target  

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Iris dataset loaded and normalized. Shape:", X_scaled.shape)

latent_dim = 5  

#reparameterization trick  z = neu+sigma.epsilon
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#encoder model
inputs = keras.Input(shape=(4,)) 
x = layers.Dense(16, activation="relu")(inputs)
x = layers.Dense(8, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
print("✅ Encoder Model Created.")

#decoder model
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8, activation="relu")(latent_inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(4, activation="linear")(x)

decoder = keras.Model(latent_inputs, outputs, name="decoder")
print("✅ Decoder Model Created.")

outputs = decoder(z)

#vae loss KL Divergence ()
class VAELossLayer(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        self.add_loss(kl_loss * 0.01)
        return inputs

vae_loss_layer = VAELossLayer()([z_mean, z_log_var])

#vae model
vae = keras.Model(inputs, outputs, name="vae")

def vae_loss(y_true, y_pred):
    reconstruction_loss = K.mean(K.square(y_true - y_pred)) * 4  
    return reconstruction_loss

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=vae_loss)
print("✅ VAE Model Compiled.")

vae.fit(X_scaled, X_scaled, epochs=300, batch_size=16, verbose=1)
print("✅ VAE Training Completed.")

num_samples = 200  
real_latents, _, _ = encoder.predict(X_scaled)

alpha = np.linspace(0, 1, num_samples)[:, np.newaxis]
random_latents = (1 - alpha) * real_latents[np.random.randint(0, len(real_latents), num_samples)] + \
                 alpha * real_latents[np.random.randint(0, len(real_latents), num_samples)]

synthetic_data = decoder.predict(random_latents)

synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)

synthetic_df = pd.DataFrame(synthetic_data_rescaled, columns=iris.feature_names)
print("✅ Synthetic Data Generated.")
print(synthetic_df.head())

real_df = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=iris.feature_names)

plt.figure(figsize=(12, 6))
for i, col in enumerate(real_df.columns):
    plt.subplot(2, 2, i + 1)
    sns.kdeplot(real_df[col], label="Real", fill=True, alpha=0.5)
    sns.kdeplot(synthetic_df[col], label="Synthetic", fill=True, alpha=0.5)
    plt.title(col)
    plt.legend()
plt.tight_layout()
plt.show()

print(synthetic_df)

encoder.save("vae_encoder_model.keras")
decoder.save("vae_decoder_model.keras")
vae.save("vae_full_model.keras")

print("✅ Models saved.")

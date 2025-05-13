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

# Load Iris dataset
iris = load_iris()
X = iris.data  
y = iris.target  

# Normalize Data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Iris dataset loaded and normalized. Shape:", X_scaled.shape)

# Latent space dimension
latent_dim = 20  

# Sampling Function (Reparameterization Trick)
def sampling(args):
    """Reparameterization trick by sampling from N(0,1)."""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# ✅ Encoder
inputs = keras.Input(shape=(4,))
x = layers.Dense(16, activation="relu")(inputs)
x = layers.Dense(8, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# Apply the Sampling Layer
z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

# Define Encoder Model
encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
print("✅ Encoder Model Created.")

# ✅ Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8, activation="relu")(latent_inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(4, activation="sigmoid")(x)  

# Define Decoder Model
decoder = keras.Model(latent_inputs, outputs, name="decoder")
print("✅ Decoder Model Created.")

# ✅ VAE Model (Combining Encoder & Decoder)
outputs = decoder(z)

# ✅ Custom KL Divergence Layer (Fixing ValueError)
class VAELossLayer(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        self.add_loss(kl_loss)  # Add KL loss to the model
        return inputs  # Just return inputs

# Apply Custom KL Divergence Layer
vae_loss_layer = VAELossLayer()([z_mean, z_log_var])

# Define the Final VAE Model
vae = keras.Model(inputs, outputs, name="vae")

# ✅ Custom VAE Loss
def vae_loss(y_true, y_pred):
    """Reconstruction loss (MSE). KL loss is automatically added."""
    reconstruction_loss = K.mean(K.square(y_true - y_pred)) * 4  
    return reconstruction_loss  # KL loss is handled by the model

# ✅ Compile the Model
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=vae_loss)
print("✅ VAE Model Compiled.")

# ✅ Train VAE
vae.fit(X_scaled, X_scaled, epochs=100, batch_size=16, verbose=1)
print("✅ VAE Training Completed.")

# ✅ Generate Synthetic Samples
num_samples = 150  
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
synthetic_data = decoder.predict(random_latent_vectors)

# ✅ Rescale Back to Original Feature Space
synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)

# ✅ Convert to DataFrame
synthetic_df = pd.DataFrame(synthetic_data_rescaled, columns=iris.feature_names)
print("✅ Synthetic Data Generated.")
print(synthetic_df.head())

# ✅ Convert Real Data to DataFrame
real_df = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=iris.feature_names)

# ✅ Plot Comparison of Distributions
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

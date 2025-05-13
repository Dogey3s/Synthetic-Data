import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from tensorflow import keras
from keras import layers
from tensorflow.keras import backend as K

# === Directory setup ===
SAVE_DIR = r"P:\synthdata\iris\plots"
CHECKPOINT_DIR = r"P:\synthdata\iris\checkpoints"
MODEL_DIR = r"P:\synthdata\iris\final_models"
DATA_DIR = r"P:\synthdata\iris\synthetic_output"
SCALER_DIR = r"P:\synthdata\iris\scalersscalers"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# === Load and preprocess Iris dataset ===
iris = load_iris()
X = iris.data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, f"{SCALER_DIR}/iris_scaler.save")
print("✅ Iris dataset normalized and scaler saved. Shape:", X_scaled.shape)

latent_dim = 5

# === Reparameterization trick ===
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# === Encoder ===
inputs = keras.Input(shape=(4,))
x = layers.Dense(16, activation="relu")(inputs)
x = layers.Dense(8, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
print("✅ Encoder Model Created.")

# === Decoder ===
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8, activation="relu")(latent_inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(4, activation="linear")(x)
decoder = keras.Model(latent_inputs, outputs, name="decoder")
print("✅ Decoder Model Created.")

# === VAE Model ===
outputs = decoder(z)

class VAELossLayer(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(kl_loss * 0.01)
        return inputs

vae_loss_layer = VAELossLayer()([z_mean, z_log_var])
vae = keras.Model(inputs, outputs, name="vae")

def vae_loss(y_true, y_pred):
    reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred)) * 4
    return reconstruction_loss

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=vae_loss)
print("✅ VAE Model Compiled.")

# === Training ===
EPOCHS = 300
BATCH_SIZE = 16
losses = []

for epoch in range(EPOCHS):
    hist = vae.fit(X_scaled, X_scaled, batch_size=BATCH_SIZE, epochs=1, verbose=0)
    loss = hist.history['loss'][0]
    losses.append(loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

    if (epoch + 1) % 50 == 0:
        encoder.save(f"{CHECKPOINT_DIR}/encoder_epoch_{epoch+1}.keras")
        decoder.save(f"{CHECKPOINT_DIR}/decoder_epoch_{epoch+1}.keras")
        vae.save(f"{CHECKPOINT_DIR}/vae_epoch_{epoch+1}.keras")

# === Plot Loss ===
plt.plot(losses)
plt.title("VAE Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(f"{SAVE_DIR}/vae_loss_plot.png")
plt.show()

# === Generate Synthetic Data ===
z_mean, _, _ = encoder.predict(X_scaled)
num_samples = 200
alpha = np.linspace(0, 1, num_samples)[:, np.newaxis]
z1 = z_mean[np.random.randint(0, len(z_mean), num_samples)]
z2 = z_mean[np.random.randint(0, len(z_mean), num_samples)]
z_synthetic = (1 - alpha) * z1 + alpha * z2
synthetic_scaled = decoder.predict(z_synthetic)

# Load and apply saved scaler for inverse transformation
scaler = joblib.load(f"{SCALER_DIR}/iris_scaler.save")
synthetic_data = scaler.inverse_transform(synthetic_scaled)
synthetic_df = pd.DataFrame(synthetic_data, columns=iris.feature_names)

# === Save synthetic data ===
synthetic_df.to_csv(f"{DATA_DIR}/iris_synthetic_data.csv", index=False)
print("✅ Synthetic data saved to iris_synthetic_data.csv")

# === KDE Plots ===
real_df = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=iris.feature_names)
plt.figure(figsize=(12, 6))
for i, col in enumerate(real_df.columns):
    plt.subplot(2, 2, i + 1)
    sns.kdeplot(real_df[col], label="Real", fill=True, alpha=0.5)
    sns.kdeplot(synthetic_df[col], label="Synthetic", fill=True, alpha=0.5)
    plt.title(col)
    plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/synthetic_vs_real_kde.png")
plt.show()

# === Final Model Saves ===
encoder.save(f"{MODEL_DIR}/vae_encoder_model.keras")
decoder.save(f"{MODEL_DIR}/vae_decoder_model.keras")
vae.save(f"{MODEL_DIR}/vae_full_model.keras")
print("✅ Final models and scaler saved.")

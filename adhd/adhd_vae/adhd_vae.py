import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers
from tensorflow.keras import backend as K
from scipy.stats import zscore
import joblib

# === Directory Setup ===
OUTPUT_DIR = r"P:\synthdata\adhd\adhd_outputs"
MODEL_DIR = r"P:\synthdata\adhd\adhd_models"
SCALER_DIR = r"P:\synthdata\adhd\scalers"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# === Load and Clean Dataset ===
df = pd.read_csv("P:/DataSets/allSubs_testSet_phenotypic_dx.csv")
features = ["Age", "ADHD Index", "Inattentive", "Hyper/Impulsive", "Full4 IQ"]

# Replace invalid entries with NaN
invalid_entries = ["-999", "-999.0", "N/A", "pending", "L", ""]
df = df[features].replace(invalid_entries, np.nan).dropna()
df = df.astype(float)
df = df[(np.abs(zscore(df)) < 3).all(axis=1)]  # Remove outliers

# === Normalize Data ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)
joblib.dump(scaler, os.path.join(SCALER_DIR, "adhd_scaler.save"))
print("✅ Scaler saved.")

# === Hyperparameters ===
input_dim = X_scaled.shape[1]
latent_dim = 5

# === Sampling Function ===
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# === Encoder ===
inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(16, activation="relu")(inputs)
x = layers.Dense(8, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

# === Decoder ===
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8, activation="relu")(latent_inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(input_dim, activation="linear")(x)
decoder = keras.Model(latent_inputs, outputs, name="decoder")

# === VAE Model ===
outputs = decoder(z)
class VAELossLayer(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        self.add_loss(kl_loss * 0.01)
        return inputs
vae_loss_layer = VAELossLayer()([z_mean, z_log_var])
vae = keras.Model(inputs, outputs, name="vae")

# === Loss Function and Compile ===
def vae_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred)) * input_dim

vae.compile(optimizer=keras.optimizers.Adam(0.001), loss=vae_loss)
vae.fit(X_scaled, X_scaled, epochs=300, batch_size=16, verbose=1)

# === Generate Synthetic Data ===
num_samples = 200
real_latents, _, _ = encoder.predict(X_scaled)
alpha = np.linspace(0, 1, num_samples)[:, np.newaxis]
z1 = real_latents[np.random.randint(0, len(real_latents), num_samples)]
z2 = real_latents[np.random.randint(0, len(real_latents), num_samples)]
z_mix = (1 - alpha) * z1 + alpha * z2
synthetic_scaled = decoder.predict(z_mix)
synthetic_data = scaler.inverse_transform(synthetic_scaled)

# === Save Outputs ===
real_df = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=features)
synthetic_df = pd.DataFrame(synthetic_data, columns=features)
synthetic_df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_adhd_data.csv"), index=False)

# === KDE Plots ===
plt.figure(figsize=(12, 6))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.kdeplot(real_df[col], label="Real", fill=True, alpha=0.5)
    sns.kdeplot(synthetic_df[col], label="Generated", fill=True, alpha=0.5)
    plt.title(col)
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "real_vs_synthetic_kde.png"))
plt.show()

# === Save Models ===
encoder.save(os.path.join(MODEL_DIR, "vae_encoder.keras"))
decoder.save(os.path.join(MODEL_DIR, "vae_decoder.keras"))
vae.save(os.path.join(MODEL_DIR, "vae_full.keras"))

print("✅ Synthetic ADHD data generation complete.")

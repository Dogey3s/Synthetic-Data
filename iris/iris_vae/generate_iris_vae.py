# === generate_iris_vae.py ===
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.datasets import load_iris
from keras.saving import register_keras_serializable

# === Register the sampling function for deserialization ===
@register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# === Paths ===
ENCODER_PATH = r"P:\synthdata\iris\final_models\vae_encoder_model.keras"
DECODER_PATH = r"P:\synthdata\iris\final_models\vae_decoder_model.keras"
SCALER_PATH = r"P:\synthdata\iris\scalers\iris_scaler.save"
OUTPUT_DIR = r"P:\synthdata\iris\synthetic_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load models ===
encoder = tf.keras.models.load_model(ENCODER_PATH, compile=False)
decoder = tf.keras.models.load_model(DECODER_PATH, compile=False)
print("✅ Encoder and Decoder loaded.")

# === Load saved scaler ===
scaler = joblib.load(SCALER_PATH)
print("✅ Scaler loaded.")

# === Load and scale original Iris data ===
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
X_scaled = scaler.transform(X)

# === Sample latent vectors from real encodings ===
z_mean, _, _ = encoder.predict(X_scaled)

# === Generate synthetic latent vectors using interpolation ===
num_samples = 200
alpha = np.linspace(0, 1, num_samples)[:, np.newaxis]
z1 = z_mean[np.random.randint(0, len(z_mean), num_samples)]
z2 = z_mean[np.random.randint(0, len(z_mean), num_samples)]
z_synthetic = (1 - alpha) * z1 + alpha * z2

# === Decode latent vectors ===
synthetic_scaled = decoder.predict(z_synthetic)
synthetic_data = scaler.inverse_transform(synthetic_scaled)

# === Save synthetic data ===
df_synthetic = pd.DataFrame(synthetic_data, columns=iris.feature_names)
csv_path = os.path.join(OUTPUT_DIR, "generated_synthetic_iris.csv")
df_synthetic.to_csv(csv_path, index=False)

print(f"✅ Generated {len(df_synthetic)} synthetic Iris records.")
print(f"📁 Saved to: {csv_path}")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset and scale it
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data                     #.venv\scripts\Activate.ps1
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

# Load the trained models with safe_mode=False to allow Lambda layer deserialization
encoder = keras.models.load_model('vae_encoder_model.keras', custom_objects={'sampling': sampling}, compile=False, safe_mode=False)

decoder = keras.models.load_model('vae_decoder_model.keras', compile=False)

# Get real latent representations
real_latents, _, _ = encoder.predict(X_scaled)

# Generate synthetic latents by interpolating between two randomly chosen real latent points
num_samples = 200
alpha = np.linspace(0, 1, num_samples)[:, np.newaxis]
random_latents = (1 - alpha) * real_latents[np.random.randint(0, len(real_latents), num_samples)] + \
                 alpha * real_latents[np.random.randint(0, len(real_latents), num_samples)]

# Decode the synthetic latents to generate synthetic data
synthetic_data = decoder.predict(random_latents)

# Rescale the synthetic data back to the original feature space
synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)

# Create DataFrame for synthetic data
synthetic_df = pd.DataFrame(synthetic_data_rescaled, columns=iris.feature_names)
print("✅ Synthetic Data Generated.")
print(synthetic_df.head())

# Visualize real vs synthetic distributions
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

# Optionally, save the synthetic dataset for later use
synthetic_df.to_csv("synthetic_data.csv", index=False)
print("✅ Synthetic data saved to synthetic_data.csv")

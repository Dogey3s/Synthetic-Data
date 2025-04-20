import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K  # Changed import to avoid tf.python internal reference

# Custom sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Custom KL loss layer
class VAELossLayer(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        self.add_loss(kl_loss * 0.01)
        return inputs

# Custom reconstruction loss
def vae_loss(y_true, y_pred):
    reconstruction_loss = K.mean(K.square(y_true - y_pred)) * 4
    return reconstruction_loss

# ✅ Load models
encoder = keras.models.load_model(
    "vae_encoder_model.keras",
    custom_objects={'sampling': sampling}
)
decoder = keras.models.load_model("vae_decoder_model.keras")
vae = keras.models.load_model(
    "vae_full_model.keras",
    custom_objects={
        'sampling': sampling,
        'VAELossLayer': VAELossLayer,
        'vae_loss': vae_loss
    }
)

print("✅ Models loaded successfully.")

# Load and normalize the iris dataset
iris = load_iris()
X = iris.data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Encode real data to latent space
real_latents, _, _ = encoder.predict(X_scaled)

# Interpolate between latent points
num_samples = 200
alpha = np.linspace(0, 1, num_samples)[:, np.newaxis]
latents1 = real_latents[np.random.randint(0, len(real_latents), num_samples)]
latents2 = real_latents[np.random.randint(0, len(real_latents), num_samples)]
random_latents = (1 - alpha) * latents1 + alpha * latents2

# Decode the latent vectors
synthetic_data = decoder.predict(random_latents)

# Rescale to original range
synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)

# Create DataFrame
synthetic_df = pd.DataFrame(synthetic_data_rescaled, columns=iris.feature_names)

print("✅ Synthetic data generated from loaded model:")
print(synthetic_df.head())

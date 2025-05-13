import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load original scaler and features
features = ["Age", "ADHD Index", "Inattentive", "Hyper/Impulsive", "Full4 IQ"]

# Load the original dataset
original_df = pd.read_csv(r"P:\DataSets\allSubs_testSet_phenotypic_dx.csv")

# Filter and clean only selected numeric columns
numeric_df = original_df[features].copy()
numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
numeric_df.dropna(inplace=True)

# Fit scaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(numeric_df)

# Load decoder
decoder = keras.models.load_model("adhd_models/vae_decoder.keras")

# Sample latent space
num_samples = 200
latent_dim = decoder.input_shape[1]
random_latents = np.random.normal(0, 1, size=(num_samples, latent_dim))
synthetic_scaled = decoder.predict(random_latents)
synthetic_data = scaler.inverse_transform(synthetic_scaled)
synthetic_df = pd.DataFrame(synthetic_data, columns=features)

# Save synthetic samples
synthetic_df.to_csv("adhd_outputs/generated_synthetic_adhd.csv", index=False)
print("✅ Generated synthetic ADHD samples saved.")

# Optional: plot comparison
real_df = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=features)
plt.figure(figsize=(12, 6))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.kdeplot(real_df[col], label="Real", fill=True)
    sns.kdeplot(synthetic_df[col], label="Generated", fill=True)
    plt.title(col)
    plt.legend()
plt.tight_layout()
plt.savefig("adhd_outputs/generated_vs_real_kde.png")
plt.show()

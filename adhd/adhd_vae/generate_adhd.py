import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# === Define paths ===
SCALER_PATH = r"P:\synthdata\adhd\scalers\adhd_scaler.save"
DECODER_PATH = r"P:\synthdata\adhd\adhd_models\vae_decoder.keras"
CSV_OUTPUT = r"P:\synthdata\adhd\adhd_outputs\generated_synthetic_adhd.csv"
PLOT_OUTPUT = r"P:\synthdata\adhd\adhd_outputs\generated_vs_real_kde.png"
DATA_PATH = r"P:\DataSets\allSubs_testSet_phenotypic_dx.csv"

# === Load scaler and model ===
scaler = joblib.load(SCALER_PATH)
decoder = keras.models.load_model(DECODER_PATH)

# === Define features ===
features = ["Age", "ADHD Index", "Inattentive", "Hyper/Impulsive", "Full4 IQ"]

# === Clean original dataset for plotting ===
original_df = pd.read_csv(DATA_PATH)
df = original_df[features].replace(["-999", "-999.0", "N/A", "pending", "L", ""], np.nan)
df = df.dropna().astype(float)

from scipy.stats import zscore
df = df[(np.abs(zscore(df)) < 3).all(axis=1)]

X_scaled = scaler.transform(df)

# === Generate synthetic data ===
num_samples = 200
latent_dim = decoder.input_shape[1]
random_latents = np.random.normal(0, 1, size=(num_samples, latent_dim))
synthetic_scaled = decoder.predict(random_latents)
synthetic_data = scaler.inverse_transform(synthetic_scaled)

# === Save synthetic data ===
synthetic_df = pd.DataFrame(synthetic_data, columns=features)
synthetic_df.to_csv(CSV_OUTPUT, index=False)
print("✅ Generated synthetic ADHD samples saved.")

# === Plot KDE comparisons ===
real_df = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=features)
plt.figure(figsize=(12, 6))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.kdeplot(real_df[col], label="Real", fill=True, alpha=0.5)
    sns.kdeplot(synthetic_df[col], label="Generated", fill=True, alpha=0.5)
    plt.title(col)
    plt.legend()
plt.tight_layout()
plt.savefig(PLOT_OUTPUT)
plt.show()

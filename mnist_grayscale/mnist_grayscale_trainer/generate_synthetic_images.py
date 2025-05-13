import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import save_img

# === Paths ===
MODEL_PATH = r"P:\synthdata\mnsit_grayscale\mnist_final_models\generator_final.h5"
OUTPUT_DIR = r"P:\synthdata\mnsit_grayscale\synthetic_mnist"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Settings ===
NUM_IMAGES = 10000
NOISE_DIM = 128
BATCH_SIZE = 100

# === Load the trained generator ===
generator = load_model(MODEL_PATH)

# === Generate synthetic images ===
for i in range(0, NUM_IMAGES, BATCH_SIZE):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    generated_images = generator(noise, training=False)

    for j in range(BATCH_SIZE):
        index = i + j
        img = (generated_images[j] + 1.0) * 127.5  # (28, 28, 1)
        img = tf.clip_by_value(img, 0, 255)
        img = tf.cast(img, tf.uint8)
        save_img(f"{OUTPUT_DIR}/synt_mnist_{index:05d}.png", img, scale=False)

print(f"✅ {NUM_IMAGES} synthetic MNIST images saved to '{OUTPUT_DIR}'")

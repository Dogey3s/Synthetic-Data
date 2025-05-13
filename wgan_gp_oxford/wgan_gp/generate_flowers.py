import tensorflow as tf
import numpy as np
import os

# === Paths ===
MODEL_PATH = "wgan_gp_model/generator_final.h5"
OUTPUT_DIR = "synthetic_flowers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load trained generator ===
generator = tf.keras.models.load_model(MODEL_PATH)

# === Generation settings ===
NUM_IMAGES = 100
BATCH_SIZE = 100
NOISE_DIM = 100

# === Generate and save images ===
for i in range(0, NUM_IMAGES, BATCH_SIZE):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    generated_images = generator(noise, training=False)

    for j in range(BATCH_SIZE):
        index = i + j
        img = (generated_images[j] + 1) * 127.5  # [-1,1] → [0,255]
        img = tf.clip_by_value(img, 0, 255)
        img = tf.cast(img, tf.uint8)
        tf.keras.preprocessing.image.save_img(
            f"{OUTPUT_DIR}/flower_{index:05d}.png", img
        )

print(f"✅ Generated {NUM_IMAGES} synthetic flower images in '{OUTPUT_DIR}'")

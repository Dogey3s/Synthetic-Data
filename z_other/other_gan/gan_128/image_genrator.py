import os
import tensorflow as tf
import numpy as np

MODEL_PATH = "checkpoints/generator_epoch_200.h5"
OUTPUT_DIR = "synthetic_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

generator = tf.keras.models.load_model(MODEL_PATH)

NUM_IMAGES = 50
BATCH_SIZE = 100
NOISE_DIM = 100

for i in range(0, NUM_IMAGES, BATCH_SIZE):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    generated_images = generator(noise, training=False)

    for j in range(BATCH_SIZE):
        index = i + j
        img = (generated_images[j] + 1) * 127.5
        img = tf.clip_by_value(img, 0, 255)
        img = tf.cast(img, tf.uint8)
        tf.keras.preprocessing.image.save_img(
            f"{OUTPUT_DIR}/image_{index:05d}.png", img
        )

print(f"✅ Generated {NUM_IMAGES} synthetic images in '{OUTPUT_DIR}'")

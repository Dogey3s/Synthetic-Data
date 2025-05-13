import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Directory for cat images
cat_dir = r"P:\DataSets\catsvsdogs\training_set\training_set\cats"
img_size = 64
BATCH_SIZE = 64
EPOCHS = 10
NOISE_DIM = 100
SAVE_DIR = "generated_images/cats"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load & preprocess cat images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = Image.open(path).convert('RGB').resize((img_size, img_size))
            img = np.array(img) / 127.5 - 1.0
            images.append(img)
        except:
            continue
    return np.array(images)

cat_images = load_images_from_folder(cat_dir)
cat_dataset = tf.data.Dataset.from_tensor_slices(cat_images).shuffle(1000).batch(BATCH_SIZE)

# Generator model
def make_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(NOISE_DIM,)),
        layers.Dense(8*8*256, use_bias=False),
        layers.Reshape((8, 8, 256)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(3, 5, strides=2, padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Discriminator (Critic) model
def make_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(64, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# WGAN Loss Functions
def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

# Generate & Save Images
def generate_and_save_images(model, epoch, folder):
    noise = tf.random.normal([16, NOISE_DIM])
    predictions = model(noise, training=False)
    predictions = (predictions + 1.0) / 2.0  # Rescale to [0, 1]

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')
    plt.savefig(os.path.join(folder, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()

# Training Step
@tf.function
def train_step(images, generator, discriminator, g_optimizer, d_optimizer):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    for _ in range(5):  # Train discriminator more times
        with tf.GradientTape() as disc_tape:
            fake_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(fake_images, training=True)
            d_loss = discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # Clip discriminator weights
        for var in discriminator.trainable_variables:
            var.assign(tf.clip_by_value(var, -0.01, 0.01))

    with tf.GradientTape() as gen_tape:
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images, training=True)
        g_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return d_loss, g_loss

# Training Loop
def train(dataset, generator, discriminator):
    g_optimizer = tf.keras.optimizers.RMSprop(5e-5)
    d_optimizer = tf.keras.optimizers.RMSprop(5e-5)

    for epoch in range(EPOCHS):
        d_losses, g_losses = [], []
        print(f"Epoch {epoch + 1}/{EPOCHS} - CAT")

        for image_batch in tqdm(dataset):
            d_loss, g_loss = train_step(image_batch, generator, discriminator, g_optimizer, d_optimizer)
            d_losses.append(d_loss)
            g_losses.append(g_loss)

        generate_and_save_images(generator, epoch + 1, SAVE_DIR)
        print(f"Discriminator loss: {np.mean(d_losses):.4f}, Generator loss: {np.mean(g_losses):.4f}")

    generator.save(os.path.join(SAVE_DIR, "generator_cat.keras"))
    discriminator.save(os.path.join(SAVE_DIR, "discriminator_cat.keras"))

# Main Execution
cat_generator = make_generator()
cat_discriminator = make_discriminator()
train(cat_dataset, cat_generator, cat_discriminator)

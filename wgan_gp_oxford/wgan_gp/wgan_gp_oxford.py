import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory setup
SAVE_DIR = "wgan_gp_images"
CHECKPOINT_DIR = "wgan_gp_checkpoints"
MODEL_DIR = "wgan_gp_model"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load Oxford 102 Flowers dataset
(train_ds, val_ds, test_ds), ds_info = tfds.load(
    'oxford_flowers102',
    split=['train', 'validation', 'test'],
    with_info=True,
    as_supervised=True
)

# Combine all splits
full_ds = train_ds.concatenate(val_ds).concatenate(test_ds)

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, [64, 64])
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

# Apply preprocessing
full_ds = full_ds.map(lambda img, lbl: preprocess(img, lbl), num_parallel_calls=tf.data.AUTOTUNE)
BUFFER_SIZE = 8189  # Total number of images in the dataset
BATCH_SIZE = 64
train_dataset = full_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Hyperparameters
EPOCHS = 200
NOISE_DIM = 100
LAMBDA_GP = 10
CRITIC_ITER = 5

# Generator model
def make_generator():
    model = tf.keras.Sequential([
        layers.Dense(4*4*512, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((4, 4, 512)),  # → (4, 4, 512)

        layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),  # 8x8
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),  # 16x16
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),   # 32x32
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')  # 64x64
    ])
    return model


# Critic model
def make_critic():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]),
        layers.LeakyReLU(),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# Instantiate models and optimizers
generator = make_generator()
critic = make_critic()
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
critic_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

# Gradient penalty function
def gradient_penalty(real_images, fake_images):
    batch_size = tf.shape(real_images)[0]  # Get actual batch size

    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated)

    grads = tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-10)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


# Loss functions
def critic_loss(real_output, fake_output, gp, lambda_gp):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + lambda_gp * gp

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

# Save generated images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = (predictions[i] + 1) / 2  # Rescale to [0, 1]
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(f"{SAVE_DIR}/image_at_epoch_{epoch:04d}.png")
    plt.close()

# Training steps
@tf.function
def train_critic(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, NOISE_DIM])
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        real_output = critic(real_images, training=True)
        fake_output = critic(fake_images, training=True)
        gp = gradient_penalty(real_images, fake_images)
        loss = critic_loss(real_output, fake_output, gp, LAMBDA_GP)
    gradients = tape.gradient(loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
    return loss

@tf.function
def train_generator():
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        fake_output = critic(fake_images, training=True)
        loss = generator_loss(fake_output)
    gradients = tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss

# Training loop
seed = tf.random.normal([16, NOISE_DIM])

def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for real_images in dataset:
            for _ in range(CRITIC_ITER):
                _ = train_critic(real_images)
            _ = train_generator()

        # Save generated images
                # Save generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)

        # Save model checkpoints every 50 epochs
        if (epoch + 1) % 50 == 0:
            generator.save(os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch+1}.h5"))
            critic.save(os.path.join(CHECKPOINT_DIR, f"critic_epoch_{epoch+1}.h5"))

    # Save final models
    generator.save(os.path.join(MODEL_DIR, "generator_final.h5"))
    critic.save(os.path.join(MODEL_DIR, "critic_final.h5"))

# Start training
train(train_dataset, EPOCHS)

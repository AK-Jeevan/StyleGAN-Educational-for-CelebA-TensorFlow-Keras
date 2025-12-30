"""
 Title: StyleGAN — CelebA

Description:
This file contains a compact, educational StyleGAN-style implementation
that demonstrates the core ideas: a mapping network (z → w), style
modulation, per-channel noise injection, a learned constant input, and a
convolutional discriminator. It is intentionally minimal and designed for
learning and experimentation on small image sizes (e.g., CelebA).

Key Steps:
1. Load and preprocess CelebA images to the [-1, 1] range used by the generator.
2. Define a mapping network to transform latent `z` into style `w`.
3. Implement style modulation and noise injection layers used in StyleGAN.
4. Build a generator that starts from a learned constant and applies styles.
5. Build a simple discriminator and standard GAN training loop.

Purpose:
- Educational reference for StyleGAN building blocks rather than a full
  production implementation. Useful for experimentation and small-scale runs.

Frameworks:
- TensorFlow 2.x / Keras
- TensorFlow Datasets (CelebA)
"""

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Hyperparameters and constants
# ------------------------------
IMG_SIZE = 64
BATCH_SIZE = 32
LATENT_DIM = 128

# ------------------------------
# Dataset: CelebA loading and preprocessing
# ------------------------------
def preprocess(x):
    # Resize images to target resolution and scale to [-1, 1]
    img = tf.image.resize(x["image"], (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img

dataset = tfds.load("celeb_a", split="train")
dataset = dataset.map(preprocess).shuffle(10000).batch(BATCH_SIZE)

# ------------------------------
# Mapping network (z -> w): small MLP that produces style vectors
# ------------------------------
def mapping_network():
    z = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(256, activation="relu")(z)
    x = layers.Dense(256, activation="relu")(x)
    return tf.keras.Model(z, x, name="Mapping")

# ------------------------------
# Style modulation layer: applies affine transform from `w` to features
# ------------------------------
class StyleMod(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        # Map style vector to per-channel scaling values
        self.dense = layers.Dense(channels)

    def call(self, x, w):
        s = self.dense(w)
        s = tf.reshape(s, (-1, 1, 1, x.shape[-1]))
        # Add 1 so the layer can act as residual-style scaling
        return x * (s + 1)

# ------------------------------
# Noise injection: adds per-pixel random noise scaled by a learned weight
# ------------------------------
class NoiseInjection(layers.Layer):
    def __init__(self):
        super().__init__()
        # Single learned scalar weight per layer
        self.weight = self.add_weight(
            shape=(1,), initializer="zeros", trainable=True
        )

    def call(self, x):
        noise = tf.random.normal([tf.shape(x)[0], x.shape[1], x.shape[2], 1])
        return x + self.weight * noise

# ------------------------------
# Generator: learned constant + style-modulated conv blocks
# ------------------------------
class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Learned constant acts as the starting feature map (4x4)
        self.const = self.add_weight(
            shape=(1, 4, 4, 256),
            initializer="random_normal",
            trainable=True,
        )

        self.blocks = []
        for f in [128, 64, 32]:
            # Each block upsamples, convolves, applies style modulation,
            # injects noise, then applies an activation.
            self.blocks.append([
                layers.UpSampling2D(),
                layers.Conv2D(f, 3, padding="same"),
                StyleMod(f),
                NoiseInjection(),
                layers.LeakyReLU(0.2)
            ])

        # Convert final features to RGB image with tanh to produce [-1,1]
        self.to_rgb = layers.Conv2D(3, 1, activation="tanh")

    def call(self, w):
        x = tf.tile(self.const, [tf.shape(w)[0], 1, 1, 1])

        for up, conv, style, noise, act in self.blocks:
            x = up(x)
            x = conv(x)
            x = style(x, w)
            x = noise(x)
            x = act(x)

        return self.to_rgb(x)

# ------------------------------
# Discriminator (critic): simple conv net producing a scalar score
# ------------------------------
def discriminator():
    img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img

    for f in [32, 64, 128]:
        x = layers.Conv2D(f, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)
    return tf.keras.Model(img, out)

# ------------------------------
# Losses & optimizers: binary crossentropy GAN losses (simple baseline)
# ------------------------------
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def d_loss(real, fake):
    # Discriminator loss: real->1, fake->0
    return bce(tf.ones_like(real), real) + bce(tf.zeros_like(fake), fake)

def g_loss(fake):
    # Generator tries to make discriminator output 1 for fake images
    return bce(tf.ones_like(fake), fake)

g_opt = tf.keras.optimizers.Adam(1e-4)
d_opt = tf.keras.optimizers.Adam(1e-4)

# ------------------------------
# Build models
# ------------------------------
mapping = mapping_network()
generator = Generator()
disc = discriminator()

# ------------------------------
# Training step: compute gradients and apply optimizer updates
# ------------------------------
@tf.function
def train_step(real_images):
    z = tf.random.normal([tf.shape(real_images)[0], LATENT_DIM])

    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        w = mapping(z)
        fake_images = generator(w)

        real_out = disc(real_images)
        fake_out = disc(fake_images)

        gl = g_loss(fake_out)
        dl = d_loss(real_out, fake_out)

    g_vars = generator.trainable_variables + mapping.trainable_variables
    g_opt.apply_gradients(zip(gt.gradient(gl, g_vars), g_vars))
    d_opt.apply_gradients(zip(dt.gradient(dl, disc.trainable_variables),
                              disc.trainable_variables))
    return gl, dl

# ------------------------------
# Visualization helper: generate and show sample images
# ------------------------------
def show_images():
    z = tf.random.normal([4, LATENT_DIM])
    w = mapping(z)
    imgs = generator(w)
    imgs = (imgs + 1) / 2

    plt.figure(figsize=(4,4))
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(imgs[i])
        plt.axis("off")
    plt.show()

# ------------------------------
# Training loop (small demo run)
# ------------------------------
for epoch in range(3):
    for batch in dataset.take(100):
        gl, dl = train_step(batch)

    print(f"Epoch {epoch+1} | G:{gl:.3f} D:{dl:.3f}")
    show_images()

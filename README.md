# ✨ StyleGAN — Style-Based Generative Adversarial Network  
**TensorFlow / Keras**

This repository demonstrates a compact, educational **StyleGAN** implementation trained on the CelebA dataset using TensorFlow and Keras.

StyleGAN revolutionizes generative modeling by decoupling image generation into **style and content**. Instead of directly generating images from latent codes, a **mapping network** transforms noise into **style vectors**, which then **modulate** convolutional layers at each scale. This approach enables **unprecedented control** over generated image properties: coarse features (pose, identity), medium features (hairstyle), and fine details (texture, color).

---

## 🚀 What This Project Covers

- Loading and preprocessing **CelebA face dataset** to `[-1, 1]` range
- Building a **mapping network** (z → w) to decouple image generation
- Implementing **style modulation** via adaptive instance normalization (AdaIN)
- Implementing **per-pixel noise injection** for stochastic variation
- Building a **style-based generator** with learned constant input
- Building a **convolutional discriminator** for real/fake classification
- Binary cross-entropy GAN training loop
- Custom `tf.keras.Model` subclass for flexible generator architecture
- Image visualization utilities for generated samples
- Style vector separation for disentangled representation learning

---

## 🧠 Why Use StyleGAN?

This project helps you:

- Understand **style-based generative modeling**
- Learn about **mapping networks** for latent space disentanglement
- Master **AdaIN (Adaptive Instance Normalization)** style application
- Achieve **fine-grained control** over generated image properties
- Generate **high-quality, diverse faces** with semantic control
- Implement **stochastic variation** through noise injection
- Explore **disentangled representations** in latent space
- Apply **state-of-the-art generative modeling techniques**

StyleGAN is especially beneficial when:
- **Fine control** over image attributes is needed
- You want to separate **global structure** from **local details**
- Generating **photorealistic faces** is critical
- **Interpolation quality** between images matters
- You need **interpretable latent space** with semantic meaning

---

## 🏗️ Training Architecture

### 🔹 Mapping Network (z → w)
- **Input**: Random latent vector (`z` ~ N(0, I), dimension = 128)
- **Architecture**:
  - Dense(256) + ReLU
  - Dense(256) + ReLU
- **Output**: Style vector (`w`, dimension = 256)
- **Purpose**: 
  - Transforms input noise into **disentangled style space**
  - Prevents mode collapse by removing simple correlations
  - Enables **smooth interpolation** in style space
  - Allows comparison of images with similar styles

### 🔹 Generator (Style-Based)
- **Input**: Style vector (`w` from mapping network)
- **Architecture**:
  - **Learned constant**: 1 × 4 × 4 × 256 trainable starting feature map
  - **Block 1** (4×4 → 8×8):
    - UpSampling2D
    - Conv2D(128, 3×3)
    - **Style Modulation**: Scales features by w
    - **Noise Injection**: Adds per-pixel random variation
    - LeakyReLU(0.2)
  - **Block 2** (8×8 → 16×16):
    - UpSampling2D
    - Conv2D(64, 3×3)
    - Style Modulation
    - Noise Injection
    - LeakyReLU(0.2)
  - **Block 3** (16×16 → 32×32):
    - UpSampling2D
    - Conv2D(32, 3×3)
    - Style Modulation
    - Noise Injection
    - LeakyReLU(0.2)
  - **To RGB**: Conv2D(3, 1×1) + tanh ([-1, 1] range)
- **Output**: 64 × 64 × 3 face images
- **Key Innovation**: Style modulation replaces traditional batch normalization

### 🔹 Style Modulation (AdaIN)
- **Input**: Features [B, H, W, C] and style vector w
- **Process**:
  - Dense(C) transforms w to per-channel scaling: s = dense(w)
  - Reshape s to [B, 1, 1, C]
  - Apply scaling: features × (s + 1)
  - The +1 acts as residual scaling (allows s=0 for identity)
- **Purpose**:
  - Controls **channel-wise feature magnitudes**
  - Determines **feature statistics** at each scale
  - Enables **multi-scale style control**
  - More expressive than traditional normalization

### 🔹 Noise Injection
- **Input**: Feature maps and learnable noise weight
- **Process**:
  - Sample noise: B × H × W × 1 from normal distribution
  - Scale by learned weight: weight × noise
  - Add to features: features + (weight × noise)
- **Purpose**:
  - Introduces **stochastic variation** without affecting global structure
  - Controls fine details (skin texture, hair color)
  - Learned per-layer for optimal noise contribution
  - Prevents mode collapse

### 🔹 Discriminator
- **Input**: Image (64 × 64 × 3)
- **Architecture**:
  - Conv2D(32, 3×3, stride=2) + LeakyReLU(0.2)
  - Conv2D(64, 3×3, stride=2) + LeakyReLU(0.2)
  - Conv2D(128, 3×3, stride=2) + LeakyReLU(0.2)
  - Flatten → Dense(1)
- **Output**: Single logit for real/fake classification
- **Purpose**: Binary classification of real vs generated images

---

## 🧪 Training Strategy

- **Loss Function**: Binary Cross-Entropy (simple baseline)
- **Discriminator Loss**: 
  - Real images → target label 1
  - Generated images → target label 0
- **Generator Loss**: Tries to fool discriminator (fake → label 1)
- **Optimizer**: Adam with learning rate = 1e-4
- **Data Pipeline**: tf.data with shuffling and batching
- **Image Normalization**: [-1, 1] matching tanh output
- **Generator Variables**: Mapping network + Generator weights
- **Discriminator Variables**: Only discriminator weights

---

## 📉 Loss Functions

### Discriminator Loss (Binary Cross-Entropy)
```
L_disc = BCE(1, D(real)) + BCE(0, D(fake))
```
- Encourages D(real) → 1 and D(fake) → 0
- Standard GAN formulation

### Generator Loss
```
L_gen = BCE(1, D(G(w)))
```
- Encourages discriminator to classify generated images as real
- Simpler than min-max formulation
- Works well with style-based architecture

---

## 🔍 Key Concepts Demonstrated

- **Mapping Network**: Decoupling latent code and image generation
- **Style-Based Generation**: Per-scale style modulation via AdaIN
- **Disentangled Latent Space**: w-space more interpretable than z-space
- **Adaptive Instance Normalization (AdaIN)**: Style application mechanism
- **Noise Injection**: Stochastic detail generation
- **Learned Constant Input**: Beginning generation from learned features
- **Progressive Architecture**: Multi-scale style application
- **Custom tf.keras.Model**: Flexible generator implementation
- **Interpolation Quality**: Style mixing and smooth transitions

---

## 💾 Output Artifacts

After training, the script generates:

- **Mapping network** (`mapping`) for latent space transformation
- **Generator** (`generator`) for style-based image synthesis
- **Discriminator** (`disc`) for real/fake classification
- **Visualization samples** printed via `show_images()` function

Generated images can be created and visualized:
```python
z = tf.random.normal([batch_size, LATENT_DIM])
w = mapping(z)
images = generator(w, training=False)
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `IMG_SIZE` | 64 | Generated image resolution (square) |
| `BATCH_SIZE` | 32 | Training batch size |
| `LATENT_DIM` | 128 | Input noise dimensionality |
| `EPOCHS` | 3 | Training epochs (demo configuration) |
| `W_DIM` | 256 | Style vector dimensionality |
| `CONST_SHAPE` | (1, 4, 4, 256) | Learned constant feature map |
| Learning Rate | 1e-4 | Adam optimizer learning rate |
| Noise Init | "zeros" | Noise weight initialization |

---

## 📊 Style Modulation Details

### Why Style Modulation Over BatchNorm?

- **BatchNorm**: Normalizes across batch dimension (loses channel statistics)
- **Style Modulation**: Per-channel scaling allows full control
- **Interpretability**: Each channel can be controlled independently
- **Flexibility**: Can apply different styles at different scales

### Multi-Scale Style Control

- **Low resolution (4×4 - 8×8)**: Coarse features (pose, identity)
- **Mid resolution (16×32)**: Medium features (hairstyle, face shape)
- **High resolution (64×64)**: Fine details (skin texture, color)

### Style Mixing

Advanced technique:
```python
w1 = mapping(z1)  # Style for coarse features
w2 = mapping(z2)  # Style for fine details
# Use w1 for early blocks, w2 for later blocks
```

---

## 🔬 Noise Injection Details

### Per-Pixel Stochasticity

- Each pixel gets **independent** noise value
- Learned weight controls contribution per layer
- Prevents deterministic artifacts
- Enables detail variation without affecting global structure

### Learned Weights

- Initialized to 0 (no noise effect at start)
- Learned during training
- Different weight per block for optimal control
- Allows model to decide when noise helps

---

## ⚠️ Important Notes

- **Style Vector Dimension**: Must match mapping network output (256 here)
- **Learned Constant**: Critical innovation; replaces fixed initialization
- **Style Modulation**: Applied to **every** convolutional layer
- **Noise Injection**: Added **after** convolution, **before** activation
- **AdaIN Formula**: features × (dense(w) + 1); the +1 is essential
- **Tanh Output**: Ensures generator output in [-1, 1] range
- **Mapping Network**: Small MLP; can be scaled up for better disentanglement

---

## 📊 Expected Training Behavior

- **Epoch 1**: Discriminator easily classifies real/fake; generator learns basic structure
- **Epoch 2**: Generator improves; discriminator loss increases as fake quality improves
- **Epoch 3**: Generated faces become more realistic; noise injection adds detail variation
- **Loss Curves**: Generator loss decreases; discriminator loss stabilizes
- **Visual Quality**: Progression from blurry to increasingly realistic faces

---

## 🎯 Advanced Modifications

Potential extensions to this implementation:

- **Style mixing**: Use different w vectors for different scales
- **Truncation trick**: Truncate z sampling from standard normal for higher quality
- **Path length regularization**: Penalize sudden latent space changes
- **Progressive training**: Start with 4×4, gradually increase resolution
- **Conditioning**: Add class labels for conditional face generation
- **Inversion**: Find z codes for given real images
- **Interpolation analysis**: Smooth transitions in style space

---

## 🎨 StyleGAN Evolution

This implementation covers **StyleGAN v1** concepts:

- Mapping network (z → w)
- Style modulation (AdaIN)
- Noise injection
- Learned constant input

**StyleGAN2** improvements include:
- Redesigned architecture with better quality
- Path length regularization
- Improved normalization techniques
- Fused operations for efficiency

---

## 📜 License

MIT License

---

## ⭐ Support

If this repository helped you:

⭐ Star the repo  
🧠 Share it with other GAN and deep learning learners  
🚀 Use it as a foundation for advanced style-based generation projects  
📖 Reference it for learning about disentangled representations and AdaIN

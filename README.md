# TCG GAN V1 — Trading Card Image Generator

## Overview

**TCG GAN V1** is a PyTorch-based Generative Adversarial Network designed to synthesize Trading Card Game (TCG) images.
The project demonstrates end-to-end GAN training, including dataset acquisition, preprocessing, model architecture design, and sample visualization.

The system follows a classical adversarial paradigm where a **Generator** produces candidate images while a **Discriminator** learns to distinguish real samples from generated ones.

---

## Key Features

* Deep Convolutional GAN architecture
* Automatic Kaggle dataset integration
* GPU acceleration (CUDA support)
* Modular Generator and Discriminator implementations
* Training visualization with generated image grids
* Weight initialization aligned with DCGAN practices

---

## Repository Structure

```
.
├── tcg_gan_v1.py          # Main training pipeline
└── README.md              # Project documentation
```

---

## Dataset

The project uses a Kaggle dataset containing Trading Card images:

```
stefanneacsu/dm-01-06
```

The dataset is automatically downloaded via `kagglehub` during execution and reorganized to match the expected PyTorch `ImageFolder` structure.

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/<username>/tcg-gan-v1.git
cd tcg-gan-v1
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib tqdm kagglehub torchsummary
```

---

## Usage

Run the training pipeline:

```bash
python tcg_gan_v1.py
```

Execution flow:

1. Dataset download
2. Image preprocessing
3. DataLoader creation
4. GAN initialization
5. Adversarial training loop
6. Generated sample visualization

---

## Model Architecture

### Generator

* Latent vector projection to spatial feature map
* Progressive ConvTranspose upsampling blocks
* Batch normalization for stable training
* Final Tanh activation producing 128×128 RGB images

### Discriminator

* Multi-stage convolutional feature extraction
* Batch normalization and LeakyReLU activations
* Fully connected real/fake prediction head

---

## Training Configuration

| Parameter        | Value   |
| ---------------- | ------- |
| Image resolution | 128×128 |
| Batch size       | 128     |
| Latent dimension | 64      |
| Epochs           | 400     |
| Optimizer        | Adam    |
| Learning rate    | 0.0001  |

---

## Results

During training, generated samples are periodically displayed to monitor learning progress and mode diversity.

---

## Future Work

* Conditional GAN extension
* Wasserstein loss integration
* Model checkpointing
* Quantitative evaluation metrics (FID / IS)
* Interactive inference script
* Dataset augmentation strategies

---

## License

This project is intended for educational and research purposes.

---

## Author
Iorga Mihai Liviu

Student deep learning project exploring GAN-based image synthesis with PyTorch.

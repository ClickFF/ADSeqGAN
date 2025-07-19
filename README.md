# Auxiliary Discriminator Sequence GANs (ADSeqGAN)

ADSeqGAN is a generative adversarial network tailored for **sequence-based molecule generation** under **low-data regimes**. It is particularly suitable for generating molecules such as SMILES strings, RNA sequences, or other linear formats, especially when training data is limited and imbalanced.

## ✨ Key Features

- **Auxiliary Discriminator**: In addition to distinguishing real from generated sequences, it provides **class label feedback**, enabling better guidance and structure-aware generation.
- **Sequence-aware Generator**: Built on sequence GAN principles (e.g., SeqGAN), optimized for discrete token generation with reinforcement learning.
- **Low-Data Friendly**: Designed to work effectively in domains where positive examples (e.g., binders) are rare.
- **Class-Conditional Generation**: Enables generation of molecules specific to desired properties (e.g., RNA binders vs protein binders).

## 🛠️ Environment Setup

To install the required dependencies for this project, use the provided `environment.yml` file with Conda:

```bash
conda env create -f ADSeqGAN.yml
```

## 🚀 Usage

Train the model
```bash
# Train the model
python condi_example.py
```

You can use `.output_samples` from API to generate molecules from checkpoints.

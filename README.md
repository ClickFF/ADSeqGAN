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

`.calculate_discriptors.py` is a script to fastly choose RDKit and OpenBabel descriptors which show strongest discrimination power. Please modify Input file and output file paths to use it.

If you use ADSeqGAN in your research, please cite:

```bibtex
@article{doi:10.1021/acs.jcim.5c01737,
	journal = {Journal of Chemical Information and Modeling},
	doi = {10.1021/acs.jcim.5c01737},
	issn = {1549-9596},
	number = {19},
	publisher = {American Chemical Society (ACS)},
	title = {Auxiliary Discrminator Sequence Generative Adversarial Networks for Few Sample Molecule Generation},
	volume = {65},
	author = {Tang, Haocheng and Long, Jing and Ji, Beihong and Wang, Junmei},
	note = {[Online; accessed 2025-11-06]},
	pages = {10311--10322},
	date = {2025-09-22},
	year = {2025},
	month = {9},
	day = {22},
}
```

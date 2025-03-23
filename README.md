

# MCT-Lens: A Hybrid CNN-Transformer Model for Strong Gravitational Lens Detection

## Overview

MCT-Lens is a deep learning framework that integrates **self-supervised learning** with a **hybrid CNN-Transformer architecture** to automatically identify strong gravitational lensing systems in large-scale astronomical datasets. It leverages **MoCo (Momentum Contrastive Learning)** for pretraining and combines **CNNs (local feature extraction) and Transformers (global feature modeling)** for improved detection performance.

## Features

- **MoCo Self-Supervised Pretraining** (based on [Stein et al. 2022](https://github.com/georgestein/ssl-legacysurvey)): Trains on **3.5 million unlabeled galaxy images** to improve feature representation.
- **Hybrid CNN-Transformer Architecture:** Enhances both local and global feature extraction.
- **Distributed Training for Acceleration:** Optimized training across **four GPUs** for large-scale datasets.
- **Strong Lens Candidate Identification:** Identified **56 new strong gravitational lens candidates** in **DESI DR9**.

## Dataset

MCT-Lens was trained and evaluated using:
- **Dark Energy Spectroscopic Instrument (DESI) Legacy Survey Data Release 9 (DR9)**
- **Unlabeled Pretraining Dataset:** 3.5 million images from DESI DR9 (used for MoCo)
- **Labeled Training Data:** Previously confirmed strong lenses for fine-tuning

The model identified **56 strong lens candidates**, categorized as:
- **Grade A (14 candidates):** High-confidence strong lenses with clear Einstein rings or arcs.
- **Grade B (24 candidates):** Potential strong lenses with some uncertainties.
- **Grade C (18 candidates):** Low-confidence candidates requiring further validation.

## Installation

To run MCT-Lens, ensure you have the following dependencies installed:

```bash
pip install torch torchvision timm numpy matplotlib astropy
```

Alternatively, install all dependencies from the `requirements.txt` file:

```
pip install -r requirements.txt
```

## Usage

### 1. MoCo Pretraining (Self-Supervised Learning)

MCT-Lens employs **MoCo v2** for contrastive learning-based self-supervised pretraining. The pretraining was conducted using **Stein et al. 2022**'s framework. To access the MoCo implementation, please visit:

👉 **[Stein et al. 2022 GitHub Repository](https://github.com/georgestein/ssl-legacysurvey)**

**Pretraining details:**

- **Batch Size:** 256
- **Temperature Hyperparameter (τ):** 0.2
- **Learning Rate:** 0.015
- **Epochs:** 800
- **Distributed Training:** 4 GPUs

After pretraining, the **MCT-Lens** model was fine-tuned on labeled strong lensing datasets.

### 2. Fine-Tuning for Lens Detection

After pretraining, fine-tune the model on labeled strong lens datasets:

```
python train.py --pretrained_model checkpoint/moco.ckpt --epochs 200 --batch_size 64
```

### 3. Predicting Strong Lensing Candidates

To run inference on new galaxy images and detect strong lensing systems:

```
python predict.py --model checkpoint/mct_lens.pth --data /path/to/test/images --output results.csv
```

## Results

After applying MCT-Lens to the **DESI DR9** dataset, the model identified **56 new strong lensing candidates**, which were not included in previous catalogs (Huang et al. 2020, 2021; Stein et al. 2022). These candidates offer additional sources for verifying strong lens systems using **DESI spectra** in future studies.

- **Grade A Candidates:** 14 high-confidence lenses
- **Grade B Candidates:** 15 potential lenses
- **Grade C Candidates:** 27 low-confidence lenses

For detailed images and classifications, refer to **Figures 5 & 6** in our paper.
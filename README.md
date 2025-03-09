# AutoHash: Autoencoder-Based Compact Hashing for Scalable k-NN Search in Vector Databases

## Notes
The current release includes our trained hashing model, all datasets, and the experiment script necessary to reproduce our results. However, the full source code is part of a commercialization effort and is currently the subject of a pending patent application. Due to intellectual property considerations, we are unable to publicly release the source code at this time. We are committed to making all supplementary materials, including the source code, publicly available as soon as commercialization and patent protection are granted.
# AutoHash: Visualization and Data

## Overview
This repository provides essential supplementary materials for the **AutoHash** project, including:

- **Image Embedding Extraction**: A script that processes images using the **DINOv2** model and extracts embeddings.
- **Image Augmentation**: A script to generate augmented versions of images for dataset expansion.
- **DBLP Paper Embeddings**: A pipeline to extract research paper metadata from **DBLP** and generate dense embeddings.
- **Merging Embeddings**: A utility to merge multiple `.npy` embedding files into a single dataset.
- **Visualization Notebook**: The Jupyter notebook (`visualization.pynb`) used to generate all figures presented in our study.

---

## Code Overview

### 1. Image Embedding Extraction (`image_embedding_extraction.py`)
This script extracts feature embeddings from images using **DINOv2**. It processes a dataset of images, applies the **AutoImageProcessor**, and saves the embeddings in `.npy` format.

#### Usage
```bash
python image_embedding_extraction.py
```

Uses DINOv2 (facebook/dinov2-with-registers-giant) for feature extraction.
Supports multi-GPU execution for efficient processing.
Utilizes PyTorch DataLoader for batch processing.
Saves extracted embeddings (.npy) and corresponding image paths (.txt).
### 2. Image Augmentation (image_augmentation.py)
This script applies multiple augmentation techniques to expand the dataset, using PIL and OpenCV.

Augmentation Methods
Flipping: Random horizontal & vertical flips.
Rotation: Random rotations between -40 to +40 degrees.
Brightness/Contrast Adjustment: Random intensity changes.
Gaussian Noise: Randomly adds noise to simulate real-world distortions.
#### Usage
```bash
python image_augmentation.py
```
 
Generates multiple augmented versions of each image.
Saves augmented images in a structured folder hierarchy.
### 3. DBLP Paper Embedding Extraction (dblp_paper_embedding.py)
This script extracts metadata from DBLP XML, formats them into structured text, and converts them into dense embeddings using BGE-M3.

#### Usage
```bash
python dblp_paper_embedding.py
```
 
Parses DBLP dataset (dblp.xml) for research paper metadata (title, authors, year, venue, etc.).
Encodes the extracted text into dense embeddings using BGE-M3.
Saves embeddings in .npy format in chunked storage for large datasets.
### 4. Merging Embeddings (merge_embeddings.py)
This script merges multiple .npy files containing embeddings into a single file for efficient storage and retrieval.

#### Usage
```bash
python merge_embeddings.py
```
 
Concatenates multiple .npy embedding files into a single large NumPy array.
Handles large-scale datasets efficiently.
Supports automatic detection of .npy files in a directory.
## Contents
visualization.pynb: Jupyter notebook with step-by-step procedures for reproducing all figures from our research.


## Datasets

### AutoHash Training Dataset
- [Download Training Dataset](https://drive.google.com/drive/folders/1p09OFWosYdZy9dIpE-syiH2hhCaN2h7V?usp=sharing)

### Additional Datasets
The following datasets were used to benchmark AutoHash’s performance:

- **iNaturalist 2018**: [Download](https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz)
- **Google Landmark Dataset v2 (GLDv2)**: [Download](https://github.com/cvdfoundation/google-landmark)
- **Rp2k Dataset**: [Download](https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/rp2k_dataset.zip)

### DBLP Dataset
AutoHash also utilizes the **DBLP dataset** for evaluations. The dataset is available in XML format and can be downloaded as follows:

- **DBLP XML Dataset**: [Download](https://dblp.uni-trier.de/xml/dblp.xml.gz)
- Download using the command:
  ```bash
  wget https://dblp.uni-trier.de/xml/dblp.xml.gz
  ```
## How to Use
1. **Clone the repository:**
   ```bash
   git clone <repository-link>
   ```
2. **Download the training dataset:**
   - [Training dataset link](https://drive.google.com/drive/folders/1p09OFWosYdZy9dIpE-syiH2hhCaN2h7V?usp=sharing)
   - Place the dataset within the cloned repository directory for easy access.

3. **Run the Visualization Notebook:**
   - Open `visualization.pynb` in Jupyter Notebook or JupyterLab.
   ```bash
   jupyter notebook visualization.pynb
   ```
   - Follow notebook instructions to generate figures.


- The full source code for AutoHash remains pending due to patent and commercialization efforts. It will be made publicly available upon completion of these processes.

## Citation
If you utilize the dataset or visualization notebook, please cite our work as follows:

 

## Contact
For inquiries, please contact:
- **Xiaowei Xu**  
- **Email:** [xwxu@ualr.edu]
- **Xingqiao Wang**  
- **Email:** [xwang1@ualr.edu]
- **Zi Wang**  
- **Email:** [zwang@ualr.edu]

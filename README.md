# AutoHash: Autoencoder-Based Compact Hashing for Scalable k-NN Search in Vector Databases
## Note:
The current release includes our trained hashing model, all datasets, and the experiment script necessary to reproduce our results. However, the full source code is part of a commercialization effort and is currently the subject of a pending patent application. Due to intellectual property considerations, we are unable to publicly release the source code at this time. We are committed to making all supplementary materials, including the source code, publicly available as soon as commercialization and patent protection are granted.

# AutoHash: Running the Indexing Experiment

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.7+** (Recommended)

Required Python Libraries: Install all necessary Python packages by running these commands in your Colab notebook or terminal:
``` bash
pip install numpy  
pip install torch        # If you are using torch  
pip install simsimd      # If you are using simsimd  
pip install faiss-cpu    # For CPU support, or use pip install faiss-gpu if you have a compatible GPU runtime  
pip install matplotlib
```
## Step 1: Modify Your Dataset Path in Config Files
Prepare datasets and modify the Dataset path in the Config Files.
## Step 2: Generate Experiment Plan (First Run)
Open your terminal or command prompt, navigate to the **image_experiment/** directory.

Execute the main script to generate the experiment plan:
```bash
python image_index_evaluation.py
```
Under the experiment plan, get the **vector_counts** for your dataset, then modify the **vector_counts** in **index_evaluation.json** to match.

## Step 3: Run Experiment Evaluation (Second Run)

In your terminal, from the **image_experiment/** directory, execute **image_index_evaluation.py** again:
```bash
python image_index_evaluation.py
```
The result will be generated and saved to the experiment plan folder. 

(Optional) Continue to a visualization process.

# Datasets
## AutoHash Dataset - S3 Download Guide

This repository provides step-by-step instructions on how to download datasets stored in an AWS S3 bucket.

## 📌 Dataset Information
The datasets are hosted in an **AWS S3 bucket** and include embeddings for various datasets such as Google Landmark, iNaturalist, RP2K, and DBLP.

### **📍 Dataset Paths**
| Dataset | S3 Path |
|---------|---------|
| Google Landmark 2 (Augmented - Index) | `s3://autohashdataset/dataset/image/google landmark 2 augmentation/embeddings_optimized_index_augmented_full.npy` |
| Google Landmark 2 (Augmented - Test) | `s3://autohashdataset/dataset/image/google landmark 2 augmentation/embeddings_optimized_test_augmented_full.npy` |
| Google Landmark 2 (Full Embeddings) | `s3://autohashdataset/dataset/image/google landmark 2/embeddings_optimized_full.npy` |
| Google Landmark 2 (Index) | `s3://autohashdataset/dataset/image/google landmark 2/embeddings_optimized_index_full.npy` |
| Google Landmark 2 (Test) | `s3://autohashdataset/dataset/image/google landmark 2/embeddings_optimized_test_full.npy` |
| iNaturalist (Train/Val 2018) | `s3://autohashdataset/dataset/image/iNaturalist/train_val_2018.npy` |
| RP2K | `s3://autohashdataset/dataset/image/rp2k/rp2k.npy` |
| DBLP Embeddings | `s3://autohashdataset/dataset/dblp/dblp_embeddings.npy` |

---

## 🚀 **Step-by-Step Guide to Download the Dataset**
### **1️⃣ Install AWS CLI**
First, install the **AWS Command Line Interface (CLI)** if you haven’t already.

- **Windows**: Download and install from [AWS CLI Installer](https://aws.amazon.com/cli/)
- **Mac (Homebrew)**:  
  ```sh
  brew install awscli
  ```
  - **Linux**
   ```sh
  sudo apt install awscli  # Debian/Ubuntu
  sudo yum install awscli  # CentOS/RHEL
  ```
### **2️⃣ Get AWS Access Credentials**
To access the dataset, you need AWS credentials (Access Key and Secret Key). Follow these steps:

Log in to AWS Console:[https://aws.amazon.com/cli/](https://console.aws.amazon.com/iam/) 
Navigate to IAM → Users → Select your user.
Click Security Credentials → Create Access Key.
Copy your Access Key ID and Secret Access Key (you won’t see the secret key again after closing the window).
### **3️⃣ Configure AWS CLI**
```sh
aws configure
```
You will be prompted to enter:
```sh
AWS Access Key ID [None]: YOUR_ACCESS_KEY_ID
AWS Secret Access Key [None]: YOUR_SECRET_ACCESS_KEY
Default region name [None]: us-east-2
Default output format [None]: 
```
### **4️⃣ Download the Dataset**
Use the following command to download a dataset file:
```sh
aws s3 cp "s3://autohashdataset/dataset/image/google landmark 2/embeddings_optimized_full.npy" .
```
To download all datasets, run:
```sh
aws s3 cp --recursive s3://autohashdataset/dataset/ .
```

## AutoHash  Sub-Dataset - Google Drive Download
### AutoHash Training Dataset
This is a partial dataset's link:
- [Download Partial Dataset from Google Drive](https://drive.google.com/drive/folders/1p09OFWosYdZy9dIpE-syiH2hhCaN2h7V?usp=sharing)

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



## Data process Overview
This repository provides essential supplementary materials for the **AutoHash** project, including:

- **Image Embedding Extraction**: A script that processes images using the **DINOv2** model and extracts embeddings.
- **Image Augmentation**: A script to generate augmented versions of images for dataset expansion.
- **DBLP Paper Embeddings**: A pipeline to extract research paper metadata from **DBLP** and generate dense embeddings.
- **Merging Embeddings**: A utility to merge multiple `.npy` embedding files into a single dataset.
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

# AutoHash: Visualization
- **Visualization Notebook**: The Jupyter notebook (`visualization.pynb`) used to generate all figures presented in our study.
**Run the Visualization Notebook:**
   - Open `visualization.pynb` in Jupyter Notebook or JupyterLab.
   ```bash
   jupyter notebook visualization.pynb
   ```
   - Follow notebook instructions to generate figures.

## Citation
If you utilize AutoHash, please cite our work.

 

## Contact
For inquiries, please contact:
- **Xiaowei Xu**  
- **Email:** [xwxu@ualr.edu]
- **Xingqiao Wang**  
- **Email:** [xwang1@ualr.edu]
- **Zi Wang**  
- **Email:** [zwang@ualr.edu]

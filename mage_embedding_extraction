import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader

# Load the DINOv2 processor and model
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-giant')
model = AutoModel.from_pretrained('facebook/dinov2-with-registers-giant')

# Move model to GPU(s)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = torch.nn.DataParallel(model)
model = model.to(device)

class ImageDataset(Dataset):
    """Custom Dataset for loading and preprocessing images."""
    def __init__(self, image_paths, image_size=(224, 224)):
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Loads and preprocesses an image."""
        image_path = self.image_paths[idx]
        try:
            # Open and resize the image
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.image_size)  # Resize to fixed dimensions
            return image_path, image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return image_path, None

def collate_fn(batch):
    """Custom collate function to handle None images."""
    valid_items = [(path, img) for path, img in batch if img is not None]
    paths, images = zip(*valid_items)
    inputs = processor(images=list(images), return_tensors="pt")  # Removed padding=True
    return paths, inputs


def collect_image_paths(dataset_dir, max_images=None):
    """
    Collects image paths from the dataset directory, up to a maximum number if specified.
    
    Parameters:
    - dataset_dir (str): Path to the dataset directory.
    - max_images (int, optional): Maximum number of image paths to collect. 
                                   If None, collects all available images.
    
    Returns:
    - list: List of collected image paths.
    """
    all_image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                all_image_paths.append(os.path.join(root, file))
                if max_images is not None and len(all_image_paths) >= max_images:
                    return all_image_paths
    return all_image_paths


def process_images_with_dataloader(image_paths, output_npy, path_file, batch_size=128, num_workers=8):
    """Process images using a DataLoader and save embeddings."""
    # Create the dataset and dataloader
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    all_embeddings = []
    all_paths = []

    # Process batches of images
    for batch_paths, batch_inputs in tqdm(dataloader, desc="Processing images"):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}  # Move inputs to GPU(s)
        with torch.no_grad():
            outputs = model(**batch_inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Append embeddings and paths
        all_embeddings.append(batch_embeddings)
        all_paths.extend(batch_paths)

    # Concatenate all embeddings into a single NumPy array
    all_embeddings = np.vstack(all_embeddings)

    # Save embeddings to a .npy file
    np.save(output_npy, all_embeddings)
    print(f"Embeddings saved to {output_npy}")

    # Save image paths to a text file
    with open(path_file, 'w') as f:
        for path in all_paths:
            f.write(path + '\n')
    print(f"Image paths saved to {path_file}")

# Define paths and parameters
dataset_path = "/scrfs/storage/xwang1/home/image_dataset/train_val_2018"  # Adjust as needed
output_npy = "/scrfs/storage/xwang1/home/image_dataset/train_val_2018.npy"
path_file = "/scrfs/storage/xwang1/home/image_dataset/train_val_2018.txt"
max_images = None  # Specify the number of images to collect
batch_size = 512  # Larger batch size for better GPU utilization
num_workers = 16  # Parallel data loading

# Step 1: Collect image paths
image_paths = collect_image_paths(dataset_path, max_images)

# Step 2: Process images using DataLoader
process_images_with_dataloader(image_paths, output_npy, path_file, batch_size=batch_size, num_workers=num_workers)

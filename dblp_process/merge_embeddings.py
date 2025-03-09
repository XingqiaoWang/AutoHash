import numpy as np
import os

def merge_numpy_files(folder_path, output_file="merged_embeddings.npy"):
    """
    Merges all .npy files in a folder into a single NumPy array and saves it.

    Args:
        folder_path (str): The path to the folder containing the .npy files.
        output_file (str): The name of the output .npy file.
    """

    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

    if not npy_files:
        print(f"No .npy files found in {folder_path}")
        return

    all_arrays = []
    for file_name in npy_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            array = np.load(file_path)
            all_arrays.append(array)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return

    if not all_arrays:
        print("No arrays were successfully loaded.")
        return

    try:
        merged_array = np.concatenate(all_arrays, axis=0)
        output_path = os.path.join(folder_path, output_file)
        np.save(output_path, merged_array)
        print(f"Merged {len(npy_files)} files into {output_path}")
    except Exception as e:
        print(f"Error merging arrays: {e}")


merge_numpy_files("/home/xwang1/pseudopeople_dataset/workspace/embedding_indexing/model_parameter_evaluation/dblp/dblp_embedding_chunks", "./dblp_embeddings.npy")
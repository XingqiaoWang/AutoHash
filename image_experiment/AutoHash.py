import numpy as np
from typing import Dict
from itertools import combinations
import random
import json
import time
import simsimd
import faiss
import torch
from indexing_model import AutoHash_Model
from torch.utils.data import DataLoader, TensorDataset

class AutoHash:
    def __init__(self, config, device ='cpu'):
        self.config = config
        model_path = config["build_index"]["model_path"]
        self.batch_size = config["build_index"]["batch_size"]
        self.margin_position = np.array(config["build_index"]["margin_position"])
        self._num_dim = int(config["build_index"]["hidden_dim"])
        vector_dim = int(config["build_index"]["vector_dim"])
        self.address_table: Dict[int, np.ndarray] = {}
        self.model = self.load_model_from_path(model_path, vector_dim, self._num_dim, device)
    
    def set_data_precision(self,precision):
        self.precision = precision
        if self.precision == 'float16':
            self.vectors = self.vectors.astype(np.float16)
        else:
            self.vectors = self.vectors.astype(np.float32)
            

    def load_model_from_path(self, model_path, input_dim, encoding_dim, device):
        """
        Create the model architecture and load the saved weights.
        """
        model = AutoHash_Model(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    def _generate_binary_addresses(self, vectors, device ='cpu'):
        row_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        embeddings = vectors / row_norms
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        dataset = TensorDataset(embeddings_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        encoded_batches = []
        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                inputs = batch[0].to(device)
                # Model returns encoded representation when second arg is True
                encoded_batch = self.model(inputs, True)
                encoded_batches.append(encoded_batch)
        encoded = torch.cat(encoded_batches, dim=0)
        
        # Use margin threshold to convert to binary patterns
        margin_position_tensor = torch.from_numpy(self.margin_position).to(device)
        binary_addresses = (encoded >= margin_position_tensor).int().cpu().numpy()
        return binary_addresses
    
    
    def generate_binary_addresses(self, vectors, cpu_to_gpu=False, device='cpu', chunk_size=3000000):
        """
        Generates binary addresses from vectors, processing in chunks on GPU with batches.

        Args:
            vectors (np.ndarray): Input vectors.
            cpu_to_gpu (bool): Whether to move data to GPU if available.
            device (str): Device to use ('cpu' or 'cuda').
            chunk_size (int): Size of chunks to process on GPU.
        Returns:
            np.ndarray: Binary addresses.
        """

        row_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        embeddings = vectors / row_norms
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

        if cpu_to_gpu and device != 'cpu':
            self.model.to(device)
            embeddings_tensor = embeddings_tensor.to(device)

        encoded_batches = []
        for i in range(0, len(embeddings_tensor), chunk_size):
            chunk = embeddings_tensor[i:i + chunk_size]

            if self.batch_size is None or self.batch_size >= len(chunk):
                # Process entire chunk at once
                with torch.no_grad():
                    encoded_batch = self.model(chunk, True).to('cpu')
                    encoded_batches.append(encoded_batch)
            else:
                # Process chunk in batches
                dataset = TensorDataset(chunk)
                data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

                with torch.no_grad():
                    chunk_encoded_batches = []
                    for batch in data_loader:
                        inputs = batch[0].to(device)
                        chunk_encoded_batches.append(self.model(inputs, True).to('cpu'))
                    encoded_batches.append(torch.cat(chunk_encoded_batches, dim=0))

        encoded = torch.cat(encoded_batches, dim=0)

        # Use margin threshold to convert to binary patterns
        margin_position_tensor = torch.from_numpy(self.margin_position).to('cpu')
        binary_addresses = (encoded >= margin_position_tensor).int().cpu().numpy()
        return binary_addresses

    def add(self, vectors, binary_addresses=None, device ='cpu'):
        if not isinstance(binary_addresses, np.ndarray):
            binary_addresses = self._generate_binary_addresses(vectors,device)
        self.vectors = vectors

        # For each row, compute the integer key and store it.
        for idx, row in enumerate(binary_addresses):
            if row.shape[0] != self._num_dim:
                raise ValueError(f"Row {idx} does not match expected dimension {self._num_dim}")
            key = self.binary_vector_to_int(row)
            self.put(key, idx)
    
    def _get_dict(self):
        return self.address_table

    def get_num_dim(self) -> int:
        """Return the number of bits (dimension) used for each address."""
        return self._num_dim

    def binary_vector_to_int(self, binary_vector: np.ndarray) -> int:
        """
        Convert a 1D numpy array of binary digits into an integer.
        
        For example, the binary vector [1, 0, 1, 0] will be converted to 10 (binary 1010).
        """
        if binary_vector.shape[0] != self._num_dim:
            raise ValueError(f"Input vector length {binary_vector.shape[0]} does not match expected {self._num_dim}")
        binary_str = "".join(binary_vector.astype(str).tolist())
        return int(binary_str, 2)

    def put(self, address: int, value):
        """Add or update a value at an address (key).
        
        If the address exists, append the new value to the existing array along axis 0.
        Otherwise, create a new entry.
        """
        # Ensure the new value is at least 1D.
        new_val = np.atleast_1d(value) if not isinstance(value, np.ndarray) else np.atleast_1d(value)
        
        if address in self.address_table:
            # Ensure the existing value is also at least 1D.
            existing_val = np.atleast_1d(self.address_table[address])
            self.address_table[address] = np.concatenate([existing_val, new_val], axis=0)
        else:
            self.address_table[address] = new_val
        
    def append_to_address(self, address: int, values):
        """Append values to an existing address."""
        current = self.address_table.get(address, np.array([]))
        self.address_table[address] = np.append(current, values)

    def get_value(self, address: int) -> np.ndarray:
        """Get the stored value at an address."""
        return self.address_table.get(address, np.array([]))


    @staticmethod
    def generate_hamming_neighbors_at_distance(query_address: int, distance: int, num_bits: int) -> list:
        """
        Generate addresses at a specific Hamming distance from the query address.
        """
        neighbors = []
        for positions in combinations(range(num_bits), distance):
            new_address = query_address
            for pos in positions:
                new_address ^= (1 << pos)
            neighbors.append(new_address)
        return neighbors


    def find_by_hamming_distance(self, query_address, min_distance = 2, max_distance = 5, min_results =50, max_results = None) -> list:
        """
        Find stored addresses within a given Hamming distance from the query address.
        Returns a list of tuples: (address, distance, stored value).
        """
        # overall_start = time.time()
        matches = []
        num_bits = self.get_num_dim()
        max_results = self.max_result_num

        # Iterate over possible distances.
        for dist in range(min_distance + 1):
            # dist_start = time.time()
            candidates = self.generate_hamming_neighbors_at_distance(query_address, dist, num_bits)
            # gen_time = time.time() - dist_start
            # print(f"[Distance {dist}] Generated {len(candidates)} neighbors in {gen_time:.6f} sec")
            
            for addr in candidates:
                # Optionally, you could time each lookup individually:
                # lookup_start = time.time()
                if addr in self.address_table:
                    matches.append(self.get_value(addr))
                    if max_results is not None and len(matches) >= max_results:
                        break
            if max_results is not None and len(matches) >= max_results:
                break

        while len(matches) < min_results and dist < max_distance:
            dist = dist + 1
            candidates = self.generate_hamming_neighbors_at_distance(query_address, dist, num_bits)
            for addr in candidates:
                if addr in self.address_table:
                    matches.append(self.get_value(addr))
                    if len(matches) >= min_results:
                        break

        return np.concatenate([np.atleast_1d(x) for x in matches])
    
        
    def print_table(self):
        """Print the contents of the address table."""
        for key, value in self.address_table.items():
            print(f"Address: {key}, Value: {value}")


    def run_query(self, query, min_distance=4, max_distance=4, min_results=100, max_results: int = None) -> list:

        results = self.find_by_hamming_distance(query, min_distance,max_distance, min_results, max_results)
        return results

    
    def _get_candidate_flat(self, query_addresses):
        # overall_start = time.time()
        candidate_arrays = []
        offsets = [0]
        # query_times = []  # To record time for each run_query call
        
        for query in query_addresses:
            # start = time.time()
            candidates = self.run_query(query, min_distance=self.min_hamming, max_distance=self.max_distance, max_results=self.max_result_num)
            # query_times.append(time.time() - start)
            candidate_arrays.append(candidates)
            offsets.append(offsets[-1] + len(candidates))
        
        flat_candidates = np.concatenate(candidate_arrays, axis=0)
        offsets = np.array(offsets, dtype=np.int64)
        
        # overall_time = time.time() - overall_start
        # avg_query_time = np.mean(query_times) if query_times else 0.0
        # print(f"_get_candidate_flat overall time: {overall_time:.6f} sec, average per-query time: {avg_query_time:.6f} sec")
        
        return flat_candidates, offsets

    def process_query_vectors(self, query_vectors, device = 'cpu'):
        # transfer query_vectors to query_addrs

        binary_queries =self._generate_binary_addresses(query_vectors,device)
        if self.precision =='float16':
            query_vectors = query_vectors.astype(np.float16)
        else:
            query_vectors = query_vectors.astype(np.float32)
        
        query_addrs = []
        for idx, row in enumerate(binary_queries):
            if row.shape[0] != self._num_dim:
                raise ValueError(f"Row {idx} does not match expected dimension {self._num_dim}")
            key = self.binary_vector_to_int(row)
            query_addrs.append(key)
            
        return query_addrs, query_vectors

    def get_top_k(self, metric="cosine", result=None, k=5):
        """
        Given a 1D numpy array 'result' and a metric type, return the top k values and their indices.
        
        For "cosine", larger values are considered better, so it returns the largest k values.
        For "euclidean", smaller values are considered better, so it returns the smallest k values.
        
        Parameters:
            metric (str): Either "cosine" or "euclidean". Default is "cosine".
            result (np.ndarray): A 1D numpy array containing similarity/distance values.
            k (int): The number of top elements to return.
        
        Returns:
            tuple: (sorted_values, sorted_indices)
                - sorted_values: The k selected values sorted in the desired order.
                - sorted_indices: The corresponding indices from the original array.
        """
        if result is None:
            raise ValueError("A valid numpy array 'result' must be provided.")
        
        # Ensure result is a NumPy array
        result = np.asarray(result)
        metric = metric.lower()
        
        if metric == "euclidean":
            # For Euclidean, lower is better: get the smallest k values.
            # np.argpartition returns indices in arbitrary order.
            indices = np.argpartition(result, k)[:k]
            # Sort them in ascending order (smallest first)
            sorted_order = indices[np.argsort(result[indices])]
        elif metric == "cosine":
            # Default or cosine: higher is better: get the largest k values.
            indices = np.argpartition(result, -k)[-k:]
            # Sort them in descending order (largest first)
            sorted_order = indices[np.argsort(result[indices])[::-1]]

        return result[sorted_order], sorted_order
    def set_search_parameter(self, max_distance =4,min_hamming = 4,k=10,max_results=2000, batch_size = 1000,metric = 'cosine'):
        self.batch_size  = batch_size
        self.max_distance = max_distance
        self.k = k
        self.max_result_num = max_results
        self.metric = metric
        self.min_hamming = min_hamming
    def process_queries_in_batches(self, query_vectors, ):
        
        # overall_start = time.time()
        
        # Process queries and addresses as usual.
        query_addrs, query_vectors = self.process_query_vectors(query_vectors)
        # Convert data types based on metric.
        if self.precision == "float16":
            query_vectors = query_vectors.astype(np.float16)
        else:
            query_vectors = query_vectors.astype(np.float32)
        
        # query_vectors = query_vectors.astype(np.float32)
        
        # Dictionary to store results: key is the global query index, value is a list of tuples
        # (original candidate index, score).
        distance_list = []
        indices_list = []
        # # Timing lists for analysis.
        # batch_times = []
        # inner_times = []
        batch_size = self.batch_size
        # Process queries in batches.
        for i in range(0, len(query_addrs), batch_size):
            # batch_start = time.time()
            
            batch_query_addrs = query_addrs[i:i+batch_size]
            batch_query_vectors = query_vectors[i:i+batch_size]
            flat_candidates, offsets = self._get_candidate_flat(batch_query_addrs)
            subvectors = self.vectors[flat_candidates]
            
            for idx, query_vector in enumerate(batch_query_vectors):
                # inner_start = time.time()
                
                start_idx = offsets[idx]
                end_idx = offsets[idx+1]
                vectors2 = subvectors[start_idx:end_idx]
                
                if self.metric == "cosine":
                    # Cosine calculations use float32.
                    dist_rank = simsimd.dot(query_vector, vectors2)
                    result = np.asarray(dist_rank, dtype=np.float32)
                    # For cosine, larger values are better.
                    values, rel_indices = self.get_top_k(self.metric, result, self.k)
                elif self.metric == "euclidean":
                    # Euclidean: using float16
                    qv = query_vector  # already float16
                    dist_rank = simsimd.sqeuclidean(qv, vectors2)
                    result = np.sqrt(np.asarray(dist_rank, dtype=np.float32))
                    # For Euclidean, smaller values are better.
                    values, rel_indices = self.get_top_k(self.metric, result, self.k)
                else:
                    raise ValueError("Unsupported metric: " + self.metric)
                
                # Map relative indices from candidate pool back to original indices.
                original_indices = flat_candidates[start_idx:end_idx][rel_indices]
                
                # Build a list of (original candidate index, score) tuples.
                # query_results = [(int(orig_idx), float(score)) for orig_idx, score in zip(original_indices, values)]
                
                # # Global query index is i + idx.
                # results[i + idx] = query_results
                indices_list.append(original_indices.tolist())
                distance_list.append(values.tolist())
        
        # overall_end = time.time()
        # total_time = overall_end - overall_start
        # avg_batch_time = np.mean(batch_times) if batch_times else 0
        # avg_inner_time = np.mean(inner_times) if inner_times else 0
        
        # print("Total processing time: {:.6f} seconds".format(total_time))
        # print("Average batch processing time: {:.6f} seconds".format(avg_batch_time))
        # print("Average per-query processing time: {:.6f} seconds".format(avg_inner_time))
        
        return distance_list,indices_list

def load_config(config_path):
    """
    Load configuration from a JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_data(file_list, device = 'cpu'):
    """
    Load and preprocess real embeddings data.
    """
    def load_and_concatenate_numpy_files(file_list, axis=0):
        """
        Load multiple NumPy files from a list of file paths and concatenate them along the specified axis.

        Args:
            file_list (list of str): List of paths to .npy files.
            axis (int, optional): Axis along which to concatenate the arrays. Default is 0.

        Returns:
            np.array: The concatenated NumPy array, or None if no files were loaded.
        """
        arrays = []
        for file_path in file_list:
            try:
                arr = np.load(file_path)
                arrays.append(arr)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if arrays:
            # Concatenate along the given axis (ensuring the arrays are compatible).
            return np.concatenate(arrays, axis=axis)
        else:
            return None
    
    combined_array = load_and_concatenate_numpy_files(file_list, axis=0)
    if combined_array is not None:
        print("Combined array shape:", combined_array.shape)
    else:
        print("No arrays were loaded.")
    data = combined_array
    # row_norms = np.linalg.norm(data, axis=1, keepdims=True)
    # embeddings = data / row_norms
    # print(f"Embeddings shape: {embeddings.shape}")
    # input_dim = embeddings.shape[1]
    # embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    # dataset = TensorDataset(embeddings_tensor)
    # return dataset, input_dim, data
    return data


def search_top_k_indices(db_vectors, query_vectors, metric="cosine", top_k=5):


    # Convert input vectors to float32.
    db_vectors = db_vectors.astype(np.float32)
    query_vectors = query_vectors.astype(np.float32)
    
    if metric == "cosine":
            # Normalize database vectors.
            db_norms = np.linalg.norm(db_vectors, axis=1, keepdims=True)
            # Avoid division by zero.
            db_norms[db_norms == 0] = 1
            db_vectors = db_vectors / db_norms
            
            # Normalize query vectors.
            query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            query_norms[query_norms == 0] = 1
            query_vectors = query_vectors / query_norms

    n_db, d = db_vectors.shape
    m, d2 = query_vectors.shape
    if d != d2:
        raise ValueError("Dimension mismatch between database and query vectors.")
    
    # Build the FAISS index based on the chosen metric.
    if metric == "cosine":
        # Using inner product index. With normalized vectors, inner product equals cosine similarity.
        index = faiss.IndexFlatIP(d)
    elif metric == "euclidean":
        # Using L2 distance index; FAISS returns squared L2 distances.
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError("Unsupported metric: " + metric)
    
    # Add database vectors to the index.
    index.add(db_vectors)
    
    # Perform a batch search.
    scores, candidate_indices = index.search(query_vectors, top_k)
    
    results = {}
    for i in range(m):
        query_results = []
        for j, idx in enumerate(candidate_indices[i]):
            score = float(scores[i][j])
            if metric == "euclidean":
                # Convert squared distance to Euclidean distance.
                sqrt_score = np.sqrt(score)
                score = sqrt_score
            query_results.append((int(idx), score))
        results[i] = query_results
        
    return results


def calculate_recall(ground_truth, candidate_results):
    """
    Calculate the average recall over all queries.

    Args:
        ground_truth (dict): Dictionary mapping each query index to a list of ground truth candidate results.
                             Each ground truth result is expected to be a tuple (candidate ID, gt score).
        candidate_results (dict): Dictionary mapping each query index to a list of candidate results.
                                  Each candidate result is expected to be a tuple (candidate ID, score).

    Returns:
        float: The average recall across all queries.
    """
    recall_list = []
    for query, gt_results in ground_truth.items():
        # Build dictionaries for easier lookup.
        gt_dict = {x[0]: x[1] for x in gt_results}
        cand_results = candidate_results.get(query, [])
        cand_dict = {x[0]: x[1] for x in cand_results}
             
        # Compute the intersection of candidate IDs.
        intersection_ids = set(gt_dict.keys()).intersection(set(cand_dict.keys()))
        intersection_with_scores = [(cid, gt_dict[cid], cand_dict[cid]) for cid in intersection_ids]
        
        if len(gt_dict) > 0 and len(cand_dict) > 0:
            recall = len(intersection_ids) / len(gt_dict)
            recall_list.append(recall)
        else:
            print(f"  Query {query} skipped (empty ground truth or candidate results)")
    
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    print(f"Average Recall: {avg_recall:.6f}")
    return avg_recall

# ---------------------------------------------------------
# Example usage:
# ---------------------------------------------------------
if __name__ == '__main__':

    config_path = './configs/AutoHash_config.json'
    config = load_config(config_path)
    file_list = config['data_path']
    top_k = config['search']['top_k']
    metric = config['search']['metric']
    precision = config['search']['data_precision']
    max_hamming_distance = config['search']["max_hamming_distance"] 
    max_results_num = config['search']["max_results_num"]
    
    data  = load_data(file_list)
    if metric == 'cosine':
        row_norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = data / row_norms


    query_vectors = data[:1000]
    database = data[1000:]
    
    at = AutoHash(config)
    at.add(database)
    at.set_data_precision(precision)
    distances,indices = at.process_queries_in_batches(query_vectors,1000,top_k,max_distance = max_hamming_distance,max_results=max_results_num,metric = metric)
    gt = search_top_k_indices(database,query_vectors,top_k=top_k,metric=metric)


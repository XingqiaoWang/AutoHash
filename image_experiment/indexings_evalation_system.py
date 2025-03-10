import time
import numpy as np
import faiss
import os
import json
import glob
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from AutoHash import load_config, AutoHash

class IndexEvaluator:
    def __init__(self, config_path="config.json"):
        """Initialize evaluator with parameters from config file."""
         
        with open(config_path, 'r') as f:
            self.config = json.load(f)

            
        training_config = self.config["training"]
        self.data_path = training_config["data_path"]
        self.model_path = training_config["experiment_path"]
        self.experiment_name = self.config.get("experiment_name", "index_experiment")
        os.makedirs(self.model_path, exist_ok=True)

        # Load customizable parameters from config
        self.dim = self.config.get("dim", 1536)  # Default: 1536
        self.n_queries = self.config.get("n_queries", 1000)  # Default: 1000
        self.k = self.config.get("k", 10)  # Default: 10
        self.vector_counts = self.config.get("vector_counts", [10000, 50000, 100000, 200000, 500000])

    def set_AutoHash_config(self,config):
        self.AutoHash_config = config
    def load_and_concatenate_numpy_files(self, file_list, axis=0):
        """Load multiple NumPy files and concatenate along a given axis."""
        arrays = []
        for file_path in file_list:
            try:
                arr = np.load(file_path)
                arrays.append(arr)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if arrays:
            return np.concatenate(arrays, axis=axis)
        else:
            raise ValueError("No valid NumPy files loaded.")


    def _calculate_recall(self, predicted_scores, ground_truth_scores):
        """Compute recall while correctly handling duplicates."""
        
        total_relevant = sum(len(gt) for gt in ground_truth_scores)  # Total number of relevant items
        retrieved_relevant = 0  # Initialize relevant retrieval count

        # for i in range(len(self.queries)):
        for i in range(len(ground_truth_scores)):
            # Convert each value to a native float and round it
            gt_list = [round(float(x), 2) for x in ground_truth_scores[i]]
            pred_list = [round(float(x), 2) for x in predicted_scores[i]]
            gt_counter = Counter(gt_list)
            pred_counter = Counter(pred_list)
            
            matched_count = 0  # Track the number of correct matches

            # Match duplicates correctly
            for item in pred_counter:
                if item in gt_counter:  # Only consider valid matches
                    count_matched = min(pred_counter[item], gt_counter[item])
                    matched_count += count_matched
                    gt_counter[item] -= count_matched  # Prevent overcounting duplicates

            retrieved_relevant += matched_count  # Accumulate matches

        recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0

        return recall

    def _average_precision(self, predicted, gt):
        """Compute average precision (AP) while considering duplicates."""
        # Convert values to native floats and round them.
        predicted = [round(float(x), 2) for x in predicted]
        gt = [round(float(x), 2) for x in gt]
        
        gt_counter = Counter(gt)
        retrieved_relevant = Counter()
        hits = 0.0
        sum_precisions = 0.0

        for i, p in enumerate(predicted):
            if p in gt_counter and retrieved_relevant[p] < gt_counter[p]:
                hits += 1.0
                sum_precisions += hits / (i + 1)
                retrieved_relevant[p] += 1

        return sum_precisions / len(gt) if len(gt) > 0 else 0.0

    def _calculate_map(self, predict_scores, ground_truth_scores):
        """Compute mean average precision (mAP) over all queries."""
        return np.mean([
            self._average_precision(predict_scores[i], ground_truth_scores[i])
            for i in range(len(ground_truth_scores))
        ])

    def _dcg(self, predicted, gt):
        """Compute Discounted Cumulative Gain (DCG) while properly handling duplicates."""
        # Convert values to native floats and round them.
        predicted = [round(float(x), 2) for x in predicted]
        gt = [round(float(x), 2) for x in gt]
        
        gt_counter = Counter(gt)
        # Assign relevance scores based on the ground truth order.
        # (Assumes gt is ordered as desired; if not, sort as needed.)
        relevance_scores = {v: len(gt) - i for i, v in enumerate(gt)}
        dcg = 0.0
        for i, p in enumerate(predicted):
            if p in relevance_scores and gt_counter[p] > 0:
                dcg += relevance_scores[p] / np.log2(i + 2)
                gt_counter[p] -= 1  # Decrement so duplicates aren’t overcounted
        return dcg

    def _calculate_ndcg(self, predict_score, ground_truth_score):
        """Compute Normalized Discounted Cumulative Gain (NDCG) while handling duplicates properly."""
        ndcgs = []
        for i in range(len(ground_truth_score)):
            pred = predict_score[i]
            gt = ground_truth_score[i]
            # Convert ground truth values to native floats and round them, then sort descending.
            ideal = sorted([round(float(x), 2) for x in gt], reverse=True)
            actual_dcg = self._dcg(pred, gt)
            ideal_dcg = self._dcg(ideal, gt)
            ndcgs.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        return np.mean(ndcgs)

        
    def evaluate(self, db_vectors, query_vectors, ground_truth,gt_idx, metric, params,binary_address =None):

        # Determine index type and metric.
        index_type = params.get('index_type', 'HNSW')
        
        # Build the index. Assumes your build_index method uses the metric from params.
        index, training_time = self.build_index(db_vectors, index_type,metric, params)
        

        
        if index_type != 'AutoHash':
            # Add database vectors to the index and measure build time.
            t0 = time.time()
            index.add(db_vectors)
            build_time = time.time() - t0
            
            # Perform search on the query vectors.
            t1 = time.time()
            distances, indices = index.search(query_vectors, self.k)
            search_time = time.time() - t1
        else:    
            config = self.AutoHash_config
            precision = config['search']['data_precision']
            max_results_num = params['maximum_candidates']
            min_hamming = params['min_hamming']

            binary_address_file = np.load(binary_address)
            t0 = time.time()
            index.add(db_vectors,binary_address_file)
            build_time = time.time() - t0
            index.set_data_precision(precision)
            t1 = time.time()
            index.set_search_parameter(min_hamming=min_hamming, max_results=max_results_num,k=self.k,metric=metric)
            distances, indices  = index.process_queries_in_batches(query_vectors)
            search_time = time.time() - t1
            distances = np.array(distances)
            indices = np.array(indices)

        
        # If using LSH, recompute the distances based on the desired metric.
        if index_type == 'LSH':
            computed = np.zeros((len(query_vectors), self.k))
            for i, query in enumerate(query_vectors):
                retrieved_vectors = db_vectors[indices[i]]
                if metric == 'cosine':
                    # For cosine similarity, compute dot product (assumes vectors are normalized)
                    computed[i] = np.dot(retrieved_vectors, query)
                elif metric == 'euclidean':
                    # For euclidean, compute the Euclidean distance between the query and each retrieved vector.
                    # Note: lower distance is better.
                    computed[i] = np.linalg.norm(retrieved_vectors - query, axis=1)
                else:
                    raise ValueError("Unsupported metric for LSH. Use 'cosine' or 'euclidean'.")
            distances = np.round(computed, 2)
        
        # For non-LSH indexes, if the metric is euclidean and using an L2 index,
        # FAISS returns squared L2 distances; compute the square root then round.
        if index_type != 'LSH' and index_type != 'AutoHash' and metric == 'euclidean':
            distances = np.sqrt(distances)
            distances = np.round(distances, 2)
        recall_list =[]
        map_list = []
        ndcg_list = []
        for k in [1,2,3,5]:
        # Compute performance metrics using your helper functions.
            recall = self._calculate_recall(distances[:,:k], ground_truth[:,:k])
            map_score = self._calculate_map(distances[:,:k], ground_truth[:,:k])
            ndcg_score = self._calculate_ndcg(indices[:,:k], gt_idx[:,:k])
            recall_list.append(recall)
            map_list.append(map_score)
            ndcg_list.append(ndcg_score)
        return {
            'training_time': training_time,
            'build_time': build_time,
            'search_time': search_time,
            'top 1,2,3,5 recall': recall_list,
            'top 1,2,3,5 map': map_list,
            'top 1,2,3,5 ndcg': ndcg_list,
            'index_type': index_type,
            'n_vectors': len(db_vectors),
            'parameters': params
        }
    
    def build_index(self, data, index_type, metric, params):
        # Choose metric based on params: cosine uses inner product; euclidean uses L2.
        if metric == 'cosine':
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif metric == 'euclidean':
            faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

        training_time = 0.0  # Default training time for indexes that do not require training.

        if index_type == 'HNSW':
            M = params.get('M', 32)
            index = faiss.IndexHNSWFlat(self.dim, M, faiss_metric)
            index.hnsw.efConstruction = params.get('efConstruction', 40)
            return index, training_time
        elif index_type == 'LSH':
            nbits = params.get('nbits', 256)
            index = faiss.IndexLSH(self.dim, nbits)
            return index, training_time
        elif index_type == 'Flat':
            if metric == 'cosine':
                index = faiss.IndexFlatIP(self.dim)
            else:
                index = faiss.IndexFlatL2(self.dim)
            return index, training_time
        elif index_type == 'IVF':
            nlist = params.get('nlist', 100)
            if metric == 'cosine':
                quantizer = faiss.IndexFlatIP(self.dim)
            else:
                quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss_metric)
            # IVF requires training.
            t0 = time.time()
            index.train(data)
            training_time = time.time() - t0
            print(f"IVF training time: {training_time:.4f} seconds")
            return index, training_time
        elif index_type == 'PQ':
            pq_m = params.get('pq_m', 16)
            pq_nbits = params.get('pq_nbits', 8)
            index = faiss.IndexPQ(self.dim, pq_m, pq_nbits)
            # PQ requires training.
            t0 = time.time()
            index.train(data)
            training_time = time.time() - t0
            print(f"PQ training time: {training_time:.4f} seconds")
            return index, training_time
        elif index_type == 'AutoHash':
            index = AutoHash(self.AutoHash_config)
            return index, training_time
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def load_index(self, filepath):
        """Load a saved FAISS index."""
        if not os.path.exists(filepath):
            print(f"Index file {filepath} does not exist.")
            return None
        return faiss.read_index(filepath)


    
    def run_experiments_sequential(self, parameter_sets, experiment_plan_folder=None):
        """
        Run experiments sequentially based on the experiment plan.
        
        If an experiment_plan_folder is provided, load the experiment info from that folder
        (which includes vector_counts and other parameters). Otherwise, use self.vector_counts.
        
        Each task is defined as a tuple (n_vectors_idx, params) where n_vectors_idx
        indexes the vector_counts list, and params is one of the provided parameter sets.
        
        Parameters:
            parameter_sets (list): A list of dictionaries, where each dictionary defines a set of parameters
                for an experiment.
            experiment_plan_folder (str, optional): Path to a previously saved experiment plan folder.
                If provided, the experiment info is loaded from there.
        
        Returns:
            results (list): A list of results returned by self.evaluate for each experiment.
        """

        # If an experiment plan folder is provided, load the experiment info.
        if experiment_plan_folder is not None:
            info_file = os.path.join(experiment_plan_folder, "experiment_info.json")
            with open(info_file, "r") as f:
                experiment_info = json.load(f)
            vector_counts = experiment_info.get("vector_counts", self.vector_counts)
            print(f"Loaded vector counts from experiment plan: {vector_counts}")
        else:
            vector_counts = self.vector_counts
        
        # Create tasks based on the number of database sizes and the parameter sets.
        tasks = [(n_vectors_idx, params) for n_vectors_idx in range(len(vector_counts))
                                        for params in parameter_sets]
        
        print(f"Running {len(tasks)} experiments sequentially...", flush=True)
        results = []
        for n_vectors_idx, params in tasks:
            result = self.evaluate(n_vectors_idx, params)
            results.append(result)
        return results

    def plot_results(self, results):
        """Visualize experiment results and save to file."""
        results_by_type = defaultdict(list)
        for result in results:
            results_by_type[result['index_type']].append(result)

        style = {'HNSW': 'bo-', 'LSH': 'gs-', 'Flat': 'rD-', 'IVF': 'm^-', 'PQ': 'yv-'}

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        ax_build, ax_search, ax_recall, ax_map = axs.flatten()

        for index_type, results_list in results_by_type.items():
            n_vectors = [r['n_vectors'] for r in results_list]
            build_times = [r['build_time'] for r in results_list]
            search_times = [r['search_time'] for r in results_list]
            recalls = [r['recall'] for r in results_list]
            maps = [r['map'] for r in results_list]

            ax_build.plot(n_vectors, build_times, style.get(index_type, 'ko-'), label=index_type)
            ax_search.plot(n_vectors, search_times, style.get(index_type, 'ko-'), label=index_type)
            ax_recall.plot(n_vectors, recalls, style.get(index_type, 'ko-'), label=index_type)
            ax_map.plot(n_vectors, maps, style.get(index_type, 'ko-'), label=index_type)

        ax_build.set_title('Build Time vs. Database Size')
        ax_search.set_title('Search Time vs. Database Size')
        ax_recall.set_title('Recall vs. Database Size')
        ax_map.set_title('mAP vs. Database Size')

        for ax in [ax_build, ax_search, ax_recall, ax_map]:
            ax.set_xlabel('Number of Database Vectors')
            ax.legend()

        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = os.path.join(self.model_path, f"{self.experiment_name}_performance_{timestamp}.png")
        plt.savefig(plot_filename)
        print(f"Saved performance plot to {plot_filename}", flush=True)

    def save_results(self, results, filename=None):
        """Save experiment results to a text file with a timestamp."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = filename or os.path.join(self.model_path, f"{self.experiment_name}_results_{timestamp}.txt")
        with open(filename, 'w') as f:
            for result in results:
                f.write(f"Index type: {result['index_type']}\n")
                f.write(f"Vector count: {result['n_vectors']}\n")
                f.write(f"Parameters: {result['parameters']}\n")
                f.write(f"Build time: {result['build_time']:.2f}s\n")
                f.write(f"Search time: {result['search_time']:.4f}s\n")
                f.write(f"Recall: {result['recall']:.4f}\n")
                f.write(f"mAP: {result['map']:.4f}\n")
                f.write(f"NDCG: {result['ndcg']:.4f}\n\n")
        print(f"Saved results to {filename}", flush=True)
        

    def design_experiment_plan(self, output_folder_prefix="experiment_plan", seed=42, chunk_size = None, metrics=["cosine", "euclidean"]):
        np.random.seed(seed)

        # Create an output folder with a timestamp inside self.model_path.
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_folder = os.path.join(self.model_path, f"{output_folder_prefix}_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created experiment output folder: {output_folder}", flush=True)

        # Load the full dataset.
        print(f"Loading dataset from {self.data_path}...", flush=True)
        full_dataset = self.load_and_concatenate_numpy_files(self.data_path)
        total_vectors = full_dataset.shape[0]
        if total_vectors < self.n_queries:
            raise ValueError("Dataset does not have enough instances for queries.")

        # Save experiment information.
        experiment_info = {
            "data_path": self.data_path,
            "total_vectors": int(total_vectors),
            "n_queries": int(self.n_queries),
            "vector_counts": self.vector_counts,  # if you still want to include these counts
            "k": int(self.k),
            "metrics": metrics
        }
        info_file = os.path.join(output_folder, "experiment_info.json")
        with open(info_file, "w") as f:
            json.dump(experiment_info, f, indent=4)
        print(f"Saved experiment info to {info_file}", flush=True)

        # Prepare two versions of the data:
        # For cosine: normalized data (for dot-product based similarity)
        data_cosine = full_dataset / np.linalg.norm(full_dataset, axis=1, keepdims=True)
        # For euclidean: raw data so that distances are not affected by normalization.
        data_euclidean = full_dataset.copy()
        
        # Define the chunk size.
        if chunk_size ==None:
            chunk_size = 1000000

        # Determine merged sizes: start with 1M, then 2M, etc.
        merged_sizes = list(range(chunk_size, total_vectors + 1, chunk_size))
        if merged_sizes and merged_sizes[-1] != total_vectors:
            merged_sizes.append(total_vectors)  # ensure the full dataset is also evaluated
        else:
            merged_sizes =[total_vectors]
            
        # Dictionary to store the ground truths for each merged block and each metric.
        ground_truths = {metric: {} for metric in metrics}

        # Loop over each merged block size.
        for merged_size in merged_sizes:
            print(f"\nProcessing merged block of size {merged_size}...", flush=True)
            # Use the first merged_size indices as the current merged block.
            merged_indices = np.arange(merged_size)
            if merged_size < self.n_queries:
                raise ValueError(f"Merged block size ({merged_size}) is smaller than the required number of queries ({self.n_queries}).")
            
            # Sample query indices from the merged block.
            query_indices = np.random.choice(merged_indices, self.n_queries, replace=False)
            # Save query indices for this merged block.
            query_indices_file = os.path.join(output_folder, f"query_indices_{merged_size}.npy")
            np.save(query_indices_file, query_indices)
            print(f"Saved {self.n_queries} query indices for merged size {merged_size} to {query_indices_file}", flush=True)
            
            # Prepare query sets for each metric.
            queries = {
                "cosine": data_cosine[query_indices],
                "euclidean": data_euclidean[query_indices]
            }
            
            # The candidate pool is the merged block with query vectors excluded.
            candidate_indices = np.setdiff1d(merged_indices, query_indices)
            
            # Prepare candidate databases for each metric.
            db = {
                "cosine": data_cosine[candidate_indices],
                "euclidean": data_euclidean[candidate_indices]
            }
                     
            
            # Process each metric (cosine and euclidean) as in your original code.
            for metric in metrics:
                if metric == "cosine":
                    index = faiss.IndexFlatIP(self.dim)
                    t0 = time.time()
                    index.add(db["cosine"])
                    build_time = time.time() - t0
                    
                    t1 = time.time()
                    distances, indices = index.search(queries["cosine"], self.k)
                    search_time = time.time() - t1
                    
                    # Round cosine scores to 2 decimals.
                    rounded_distances = np.round(distances, 2)
                elif metric == "euclidean":
                    index = faiss.IndexFlatL2(self.dim)
                    t0 = time.time()
                    index.add(db["euclidean"])
                    build_time = time.time() - t0
                    
                    t1 = time.time()
                    distances, indices = index.search(queries["euclidean"], self.k)
                    search_time = time.time() - t1
                    
                    # FAISS returns squared L2 distances; compute square roots.
                    distances = np.sqrt(distances)
                    rounded_distances = np.round(distances, 2)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                # Save the computed ground truth for the current merged block.
                ground_truths[metric][str(merged_size)] = {
                    "candidate_indices": candidate_indices.tolist(),
                    "query_indices": query_indices.tolist(),
                    "ground_truth_distances": [[ "{:.2f}".format(d) for d in row ] for row in rounded_distances.tolist()],
                    "ground_truth_indices": indices.tolist(),
                    "build_time": build_time,
                    "search_time": search_time,
                }
                print(f"Computed ground truth for merged block {merged_size} using {metric}: build_time={build_time:.4f}s, search_time={search_time:.4f}s", flush=True)

        # Optionally, save the entire ground truth information.
        ground_truth_file = os.path.join(output_folder, "ground_truths.json")
        with open(ground_truth_file, "w") as f:
            json.dump(ground_truths, f, indent=4)
        print(f"\nSaved complete ground truth information to {ground_truth_file}", flush=True)

        return output_folder

    def load_experiment_results(self, output_folder):
        def _load_numpy_files(data_path, prefix):
            # Create a search pattern for files with the given prefix and .npy extension.
            pattern = os.path.join(data_path, f"{prefix}*.npy")
            file_list = sorted(glob.glob(pattern))
            if not file_list:
                raise ValueError(f"No files with prefix '{prefix}' found in {data_path}.")
            
             
            arrays = {}
            for file in file_list:
                print(f"Loading file: {file}")
                basename = os.path.basename(file)  
                rest_of_name = basename[len(prefix):] 
                rest_of_name, _ = os.path.splitext(rest_of_name)
                arr = np.load(file)
                arrays[rest_of_name]=arr
            return arrays


        gt_file = os.path.join(output_folder, "ground_truths.json")
        with open(gt_file, "r") as f:
            ground_truths = json.load(f)

        query_indices_dict = _load_numpy_files(output_folder,"query_indices_")


        info_file = os.path.join(output_folder, "experiment_info.json")
        with open(info_file, "r") as f:
            experiment_info = json.load(f)


        return ground_truths, query_indices_dict, experiment_info
    
    def create_AutoHash_binary_files_use_CUDA(self,config, database, experiment_plan_folder):
        at = AutoHash(config)
        binary_database = at.generate_binary_addresses(database, False, "cpu",1000000)
        binary_address_file = os.path.join(experiment_plan_folder, f"Binary_address_{database.shape[0]+1000}.npy")
        np.save(binary_address_file, binary_database)
        
    def evaluate_from_plan(self, distance_metrics, params, ground_truths, n_vectors, experiment_plan_folder):
        # Get the number of database vectors for this experiment.
        
        AutoHash_binary_address = os.path.join(experiment_plan_folder,f'Binary_address_{n_vectors}.npy')
        # Retrieve the saved database sample indices from the database_indices file.
        
        # Load the full dataset.
        full_dataset = self.load_and_concatenate_numpy_files(self.data_path)
        
        # Prepare two versions of the data.
        data_cosine = full_dataset / np.linalg.norm(full_dataset, axis=1, keepdims=True)
        data_euclidean = full_dataset.copy()
        
        queries = {
            "cosine": data_cosine[ground_truths["cosine"][str(n_vectors)]["query_indices"]],
            "euclidean": data_euclidean[ground_truths["euclidean"][str(n_vectors)]["query_indices"]]
        }
        # Retrieve database vectors using the saved indices.
        db = {
            "cosine": data_cosine[ground_truths["cosine"][str(n_vectors)]["candidate_indices"]],
            "euclidean": data_euclidean[ground_truths["euclidean"][str(n_vectors)]["candidate_indices"]]
        }
        
        results = {}
        for metric in distance_metrics:
            ground_truths_distance = np.array(ground_truths[metric][str(n_vectors)]["ground_truth_distances"])
            gt_idx = np.array(ground_truths[metric][str(n_vectors)]["ground_truth_indices"])
            result = self.evaluate(db[metric],queries[metric],ground_truths_distance,gt_idx, metric, params,AutoHash_binary_address)
            results[metric] = result
        return {"n_vectors": n_vectors, "params": params, "results": results}
    

    def run_experiments_from_plan(self, parameter_sets, experiment_plan_folder):
        """
        Load the experiment plan (ground truth, query indices, and database indices) from the given folder and
        run experiments sequentially using the loaded data.
        
        Parameters:
            parameter_sets (list): A list of parameter dictionaries to run experiments.
            experiment_plan_folder (str): Folder containing saved experiment info, ground truth, query indices, and database indices.
        
        Returns:
            results (list): A list of evaluation results for each experiment.
        """
        # Load the experiment plan files.
        ground_truths, query_indices_dict, exp_info = self.load_experiment_results(experiment_plan_folder)
        vector_counts = exp_info.get("vector_counts", self.vector_counts)
        distance_metrics = exp_info.get("metrics",['cosine'])
         
        # Load the full dataset and prepare query sets.
        # full_dataset = self.load_and_concatenate_numpy_files(self.data_path)
        # data_cosine = full_dataset / np.linalg.norm(full_dataset, axis=1, keepdims=True)
        # data_euclidean = full_dataset.copy()
        for count in vector_counts:
            AutoHash_binary_address = os.path.join(experiment_plan_folder,f'Binary_address_{count}.npy')
            if not os.path.exists(AutoHash_binary_address):
                full_dataset = self.load_and_concatenate_numpy_files(self.data_path)
                # Retrieve database vectors using the saved indices.
                db = full_dataset[ground_truths["euclidean"][str(count)]["candidate_indices"]]
                self.create_AutoHash_binary_files_use_CUDA(self.AutoHash_config, db,experiment_plan_folder)

        # Create tasks based on vector_counts and the parameter sets.
        tasks = [(n_idx, params) for n_idx in range(len(vector_counts)) for params in parameter_sets]
        print(f"Running {len(tasks)} experiments sequentially from plan...", flush=True)
        
        results = []
        for n_vectors_idx, params in tasks:
            result = self.evaluate_from_plan(distance_metrics, params, ground_truths, vector_counts[n_vectors_idx], experiment_plan_folder)
            results.append(result)
        return results
    
    def read_ground_truth_setup(self, experiment_folder, n_queries=5):
        """
        Read the ground truth file and query indices from the given experiment folder,
        and extract the first n_queries along with their corresponding ground truth results.
        
        For each metric (e.g., "cosine" and "euclidean") and each database size,
        this method retrieves the first n_queries ground truth indices and scores.
        
        Parameters:
            experiment_folder (str): Path to the experiment output folder.
            n_queries (int): Number of query results to extract for setup (default: 5).
        
        Returns:
            setup (dict): A dictionary summarizing the setup, including:
                - 'query_indices': List of the first n query indices.
                - For each metric, a sub-dictionary keyed by the database size (as string)
                  containing:
                    - 'ground_truth_indices': List of first n queries' ground truth indices.
                    - 'ground_truth_scores': List of first n queries' ground truth scores.
        """
        # Build file paths.
        gt_file = os.path.join(experiment_folder, "ground_truths.json")
        qi_file = os.path.join(experiment_folder, "query_indices.npy")
        
        # Load ground truth and query indices.
        with open(gt_file, "r") as f:
            ground_truths = json.load(f)
        query_indices = np.load(qi_file)
        
        # Prepare the setup dictionary.
        setup = {
            "query_indices": query_indices[:n_queries].tolist()
        }
        
        # Process each metric (e.g., "cosine", "euclidean").
        for metric, data in ground_truths.items():
            setup[metric] = {}
            for n_vectors, gt_data in data.items():
                # For each database size, extract the first n_queries results.
                setup[metric][n_vectors] = {
                    "ground_truth_indices": gt_data["ground_truth_indices"][:n_queries],
                    "ground_truth_scores": gt_data["ground_truth_distances"][:n_queries]
                }
        return setup
    

if __name__ == "__main__":
    # Update the config_path as needed
    config_path = '/scrfs/storage/xwang1/home/pseudopeople_dataset/workspace/embedding_indexing/model_parameter_evaluation/configs/traditional_index_experiment.json'
    experiment_folder = '/home/xwang1/pseudopeople_dataset/workspace/embedding_indexing/model_parameter_evaluation/traditional_index_models/experiment_plan_20250225-114435'
    evaluator = IndexEvaluator(config_path)
    # experiment_folder = evaluator.design_experiment_plan()


    #    # Define parameter sets including different index types.
    parameter_sets = [
        # # HNSW example:
        # {'index_type': 'HNSW', 'M': 64, 'efConstruction': 1600, 'efSearch': 800},
        # {'index_type': 'HNSW', 'M': 16, 'efConstruction': 400, 'efSearch': 100},
        # {'index_type': 'HNSW', 'M': 32, 'efConstruction': 400, 'efSearch': 200},
        # LSH example:
        # {'index_type': 'LSH', 'nbits': 256},
        # {'index_type': 'LSH', 'nbits': 128},
        # {'index_type': 'LSH', 'nbits': 512},
        # {'index_type': 'LSH', 'nbits': 1024},
        # {'index_type': 'LSH', 'nbits': 2048},
        # # Flat (exact search):
        # {'index_type': 'Flat'},
        # # IVF example:
        # {'index_type': 'IVF', 'nlist': 20, 'nprobe': 2},
        # {'index_type': 'IVF', 'nlist': 30, 'nprobe': 3},
        # # PQ example:
        # {'index_type': 'PQ', 'pq_m': 16, 'pq_nbits': 8}
        # AutoHash example:
        {'index_type': 'AutoHash', 'maximum_candidates':1000},
        {'index_type': 'AutoHash', 'maximum_candidates':1500},
        {'index_type': 'AutoHash', 'maximum_candidates':2000}
    ]
    experiment_folder = '/home/xwang1/pseudopeople_dataset/workspace/embedding_indexing/model_parameter_evaluation/image_index_models/experiment_plan_20250228-003244'
    results = evaluator.run_experiments_from_plan(parameter_sets, experiment_folder)
    
    # Create a timestamp string.
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Incorporate the timestamp into the filename.
    filename = f"experiment_results_{timestamp}.json"
    addr = os.path.join(experiment_folder,filename)
    # Write the results to the JSON file.
    with open(addr, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {addr}")






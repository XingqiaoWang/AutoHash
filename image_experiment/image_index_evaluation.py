from indexings_evalation_system import IndexEvaluator
import json
from AutoHash import load_config
import time
import os
if __name__ == "__main__":
    # Update the config_path as needed
    config_path = './config/index_evaluation.json'
    evaluator = IndexEvaluator(config_path)
    experiment_folder = evaluator.design_experiment_plan(chunk_size=100000)
    Auto_hash_config_path = './config/AutoHash_image_config.json'
    AutoHash_config = load_config(Auto_hash_config_path)
    evaluator.set_AutoHash_config(AutoHash_config)
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
        {'index_type': 'AutoHash', 'maximum_candidates':1000, 'min_hamming':4},
        # {'index_type': 'AutoHash', 'maximum_candidates':1500},
        # {'index_type': 'AutoHash', 'maximum_candidates':2000}
    ]

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
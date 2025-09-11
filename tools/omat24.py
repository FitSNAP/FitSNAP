from fairchem.core.datasets import AseDBDataset
import numpy as np
from collections import Counter
import multiprocessing as mp
from tqdm import tqdm
import sys

def process_chunk(args):
    """Worker function to process a chunk of the dataset"""
    path, start_idx, end_idx = args
    dataset = AseDBDataset(config=dict(src=path))
    
    species_list = []
    try:
        for i in range(start_idx, end_idx):
            species_tuple = tuple(sorted(set(dataset.get_atoms(i).get_chemical_symbols())))
            species_list.append(species_tuple)
    finally:
        # Ensure cleanup (though Python should handle this automatically)
        del dataset
    
    return species_list

def omat24elements(path):
    dataset = AseDBDataset(config=dict(src=path))
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} configurations", file=sys.stderr)
    
    # Use fewer processes to avoid "too many open files" - try 4-8 processes max
    num_cores = min(64, mp.cpu_count())  # Limit to 64 processes max
    print(f"Using {num_cores} cores", file=sys.stderr)
    
    # Calculate chunk size
    chunk_size = max(1, dataset_size // num_cores)
    
    # Create chunks
    chunks = []
    for i in range(0, dataset_size, chunk_size):
        end_idx = min(i + chunk_size, dataset_size)
        chunks.append((path, i, end_idx))
    
    print(f"Processing {len(chunks)} chunks...", file=sys.stderr)
    
    # Process chunks in parallel with progress bar
    species_list = []
    with mp.Pool(processes=num_cores) as pool:
        with tqdm(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
            for result in pool.imap_unordered(process_chunk, chunks):
                species_list.extend(result)
                pbar.update(1)
    
    print("Counting species combinations...", file=sys.stderr)
    # Count occurrences of each unique species set
    species_counter = Counter(species_list)
    
    # Filter species with >= 2 elements and sort by count (descending)
    filtered_species = [(species, count) for species, count in species_counter.items() if len(species) >= 2]
    filtered_species.sort(key=lambda x: x[1], reverse=True)  # Sort by count descending
    
    print(f"\nFound {len(filtered_species)} unique species combinations with >= 2 elements:", file=sys.stderr)
    print("(sorted by number of configs, most common first)\n", file=sys.stderr)
    
    for species_tuple, count in filtered_species:
        species_list_display = list(species_tuple)
        print(f"{count}, \"{species_list_display}\"")

if __name__ == "__main__":
    omat24elements("./rattled-1000")

from fairchem.core.datasets import AseDBDataset
import numpy as np
from collections import Counter, defaultdict
import multiprocessing as mp
from tqdm import tqdm
import sys
import pandas as pd
import os

def process_chunk(args):
    """Worker function to process a chunk of the dataset"""
    path, start_idx, end_idx, subset_name = args
    dataset = AseDBDataset(config=dict(src=path))
    
    species_list = []
    try:
        for i in range(start_idx, end_idx):
            species_tuple = tuple(sorted(set(dataset.get_atoms(i).get_chemical_symbols())))
            species_list.append((species_tuple, subset_name))
    finally:
        # Ensure cleanup (though Python should handle this automatically)
        del dataset
    
    return species_list

def process_dataset(path, subset_name):
    """Process a single dataset and return species counts"""
    print(f"Processing {subset_name}: {path}", file=sys.stderr)
    
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist, skipping {subset_name}", file=sys.stderr)
        return {}
    
    try:
        dataset = AseDBDataset(config=dict(src=path))
        dataset_size = len(dataset)
        print(f"  Dataset size: {dataset_size} configurations", file=sys.stderr)
    except Exception as e:
        print(f"Error loading dataset {subset_name}: {e}", file=sys.stderr)
        return {}
    
    num_cores = mp.cpu_count()
    
    # Calculate chunk size
    chunk_size = max(1000, dataset_size // num_cores)  # Minimum 1000 per chunk
    
    # Create chunks
    chunks = []
    for i in range(0, dataset_size, chunk_size):
        end_idx = min(i + chunk_size, dataset_size)
        chunks.append((path, i, end_idx, subset_name))
    
    print(f"  Processing {len(chunks)} chunks with {num_cores} cores...", file=sys.stderr)
    
    # Process chunks in parallel with progress bar
    species_list = []
    with mp.Pool(processes=num_cores) as pool:
        with tqdm(total=len(chunks), desc=f"  {subset_name}", unit="chunk") as pbar:
            for result in pool.imap_unordered(process_chunk, chunks):
                species_list.extend(result)
                pbar.update(1)
    
    # Count occurrences for this subset
    species_counter = Counter([species for species, _ in species_list])
    
    # Filter species with >= 2 elements
    filtered_species = {species: count for species, count in species_counter.items() if len(species) >= 2}
    
    print(f"  Found {len(filtered_species)} unique species combinations with >= 2 elements", file=sys.stderr)
    return filtered_species

def omat24elements_all_subsets():
    """Process all OMat24 subsets and create Excel output"""
    
    # Define all subsets and their paths
    subsets = {
        'rattled-300': './rattled-300',
        'rattled-500': './rattled-500', 
        'rattled-1000': './rattled-1000',
        'rattled-relax': './rattled-relax',
        'aimd-from-PBE-1000-npt': './aimd-from-PBE-1000-npt',
        'aimd-from-PBE-1000-nvt': './aimd-from-PBE-1000-nvt',
        'aimd-from-PBE-3000-npt': './aimd-from-PBE-3000-npt',
        'aimd-from-PBE-3000-nvt': './aimd-from-PBE-3000-nvt'
    }
    
    # Dictionary to store all species counts across subsets
    all_species_data = defaultdict(lambda: defaultdict(int))
    
    # Process each subset
    for subset_name, path in subsets.items():
        subset_counts = process_dataset(path, subset_name)
        
        # Add counts to the master dictionary
        for species, count in subset_counts.items():
            all_species_data[species][subset_name] = count
    
    print(f"\nCombining results from all subsets...", file=sys.stderr)
    
    # Convert to DataFrame
    data_rows = []
    for species_tuple in all_species_data.keys():
        row = {'Elements': str(list(species_tuple))}  # Convert tuple to string representation
        
        # Add counts for each subset
        total_count = 0
        for subset_name in subsets.keys():
            count = all_species_data[species_tuple][subset_name]
            row[subset_name] = count
            total_count += count
        
        row['TOTAL'] = total_count
        data_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Sort by elements alphabetically
    df = df.sort_values('Elements')
    
    # Reorder columns
    column_order = ['Elements'] + list(subsets.keys()) + ['TOTAL']
    df = df[column_order]
    
    # Fill NaN with 0
    df = df.fillna(0)
    
    # Convert counts to integers
    count_columns = list(subsets.keys()) + ['TOTAL']
    for col in count_columns:
        df[col] = df[col].astype(int)
    
    # Save to Excel file
    output_file = 'omat24_element_counts.xlsx'
    print(f"Saving results to {output_file}...", file=sys.stderr)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Element_Counts', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Element_Counts']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Max width of 50
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Results saved to {output_file}", file=sys.stderr)
    print(f"Total unique element combinations: {len(df)}", file=sys.stderr)
    print(f"Total configurations across all subsets: {df['TOTAL'].sum()}", file=sys.stderr)

if __name__ == "__main__":
    omat24elements_all_subsets()

import pandas as pd
import os
import math

# File paths
input_file = 'data/raw/all_swaps.csv'
output_dir = 'data/raw'

print("Counting total rows...")
# Get total number of rows
total_rows = sum(1 for _ in open(input_file)) - 1  # subtract 1 for header
chunk_size = math.ceil(total_rows / 2)

print(f"Total rows (excluding header): {total_rows}")
print(f"Chunk size: {chunk_size}")

print("Starting to split file...")
# Read and split the CSV in chunks
chunk_iterator = pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)

for i, chunk in enumerate(chunk_iterator, 1):
    output_file = os.path.join(output_dir, f'all_swaps_part{i}.csv')
    chunk.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(chunk)} rows")

# Verify the split
print("\nVerifying split files:")
for i in range(1, 3):
    file = os.path.join(output_dir, f'all_swaps_part{i}.csv')
    size_gb = os.path.getsize(file) / (1024**3)
    print(f"Part {i} size: {size_gb:.2f} GB") 
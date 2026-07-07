import pandas as pd
import os

# 1. Define exact paths based on your folder structure
input_file = os.path.join('archive', 'unbalanced_train_segments.csv')
output_dir = 'processed'
output_file = os.path.join(output_dir, 'filtered_vehicle_segments.csv')

# Create the processed folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 2. The AudioSet IDs for the vehicles you are classifying
target_labels = [
    '/m/0k4j',  # Car
    '/m/01bjv', # Bus
    '/m/04_sv', # Bicycle (Bike)
    '/m/07r04'  # Truck
] 

filtered_chunks = []

print(f"Scanning the massive dataset at: {input_file} ...")

# 3. Read in chunks so your computer doesn't crash from lack of memory
try:
    chunk_iterator = pd.read_csv(
        input_file, 
        comment='#', 
        skipinitialspace=True, 
        header=None, 
        names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
        quotechar='"',
        chunksize=100000
    )

    for i, chunk in enumerate(chunk_iterator):
        mask = chunk['positive_labels'].str.contains('|'.join(target_labels), na=False)
        filtered_chunks.append(chunk[mask])
        print(f"Processed 100,000 rows (Chunk {i+1})...")

    # 4. Save the final filtered list
    final_filtered_df = pd.concat(filtered_chunks, ignore_index=True)
    final_filtered_df.to_csv(output_file, index=False)

    print(f"\nSuccess! Found {len(final_filtered_df)} relevant vehicle clips.")
    print(f"Saved the filtered list to: {output_file}")

except FileNotFoundError:
    print(f"Error: Could not find {input_file}. Please ensure you are running the script from the 'traffic project' root folder!")

import pandas as pd
import numpy as np

def convert_fraction_to_float(value):
    """Convert fraction strings like '2/3' to float"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        # Check if it's a fraction like '2/3'
        if '/' in value:
            try:
                parts = value.split('/')
                return float(parts[0]) / float(parts[1])
            except:
                return np.nan
        # Try to convert to float directly
        try:
            return float(value)
        except:
            return np.nan
    return float(value)

# File paths
input_file = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/feature_paired.csv"
output_file = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/feature_paired_normalized.csv"

# Step 1: Read the aligned feature data
print("Reading feature data...")
df = pd.read_csv(input_file)
print(f"Original shape: {df.shape}")

# Step 2: Drop the last column
print("Dropping last column...")
df = df.iloc[:, :-1]
print(f"Shape after dropping last column: {df.shape}")

# Step 3: Convert all columns (except 'cell id') to numeric, handling fractions
print("Converting fraction strings to floats...")
cell_id_col = 'cell id'

for col in df.columns:
    if col == cell_id_col:
        continue
    
    # Apply fraction conversion to the column
    df[col] = df[col].apply(convert_fraction_to_float)

print("Conversion complete!")

# Step 4: Normalize each numeric column
print("Normalizing columns...")

normalized_count = 0
for col in df.columns:
    if col == cell_id_col:
        continue
    
    # Get the column data
    col_data = df[col]
    
    # Record NA positions
    na_mask = col_data.isna()
    
    # Get non-NA values
    non_na_values = col_data[~na_mask]
    
    if len(non_na_values) > 0:
        # Calculate min and max of non-NA values
        min_val = non_na_values.min()
        max_val = non_na_values.max()
        
        # Normalize: (x - min) / (max - min) + 1
        if max_val != min_val:  # Avoid division by zero
            normalized = (non_na_values - min_val) / (max_val - min_val) + 1
        else:
            # If all non-NA values are the same, set them to 1.5 (middle of [1, 2])
            normalized = pd.Series([1.5] * len(non_na_values), index=non_na_values.index)
        
        # Update the column with normalized values
        df.loc[~na_mask, col] = normalized
    
    # Set all NA values to 0
    df.loc[na_mask, col] = 0
    
    normalized_count += 1
    if normalized_count % 100 == 0:
        print(f"  Processed {normalized_count}/{len(df.columns)-1} columns...")

print(f"Normalization complete! Processed {normalized_count} columns.")

# Step 5: Save the normalized data
print(f"\nSaving to {output_file}...")
df.to_csv(output_file, index=False)

print("Done!")
print(f"Final shape: {df.shape}")
print(f"Output saved to: {output_file}")
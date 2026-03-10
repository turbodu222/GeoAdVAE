"""
Simple cell extraction script
Extract gene expression data for specified cell barcodes
"""

import pandas as pd

# 1. Read target cell barcode list (from row names)
barcode_df = pd.read_csv("/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/exon_data_top2000_name.csv", 
                         index_col=0)

# Extract cell barcodes (i.e., row names/index)
target_barcodes = barcode_df.index.tolist()

print(f"Target barcodes: {len(target_barcodes)}")
print(f"First 5 barcodes: {target_barcodes[:5]}")

# 2. Read complete expression data
expression_df = pd.read_csv("/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/exon_norm_2000.csv", 
                            index_col=0)

print(f"Expression data shape: {expression_df.shape}")

# 3. Keep only rows with target barcodes
matching_barcodes = [b for b in target_barcodes if b in expression_df.index]
filtered_data = expression_df.loc[matching_barcodes]

print(f"Matched barcodes: {len(matching_barcodes)}/{len(target_barcodes)}")
print(f"Filtered data shape: {filtered_data.shape}")

# 4. Save results
filtered_data.to_csv("/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/exon_norm_paired_2000.csv")

print(f"✓ Done! Saved to exon_norm_paired_2000.csv")
print(f"Final shape: {filtered_data.shape[0]} cells × {filtered_data.shape[1]} genes")
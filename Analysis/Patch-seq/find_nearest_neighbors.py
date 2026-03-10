import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

print("=" * 80)
print("FINDING NEAREST NEIGHBORS BETWEEN MODALITIES IN UMAP SPACE")
print("=" * 80)

# ============================================================================
# STEP 1: Load UMAP coordinates
# ============================================================================
print("\n[1] Loading UMAP coordinates...")

gex_umap_path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup23/cm_VAE_v6_PCA/outputs/attempt_1/images/coordinates/GEX_umap_coordinates_final.csv"
morpho_umap_path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup23/cm_VAE_v6_PCA/outputs/attempt_1/images/coordinates/Morpho_umap_coordinates_final.csv"

# Read UMAP coordinates
gex_umap = pd.read_csv(gex_umap_path)
morpho_umap = pd.read_csv(morpho_umap_path)

print(f"   GEX UMAP shape: {gex_umap.shape}")
print(f"   GEX columns: {gex_umap.columns.tolist()}")
print(f"   Sample GEX data:\n{gex_umap.head(3)}")

print(f"\n   Morphology UMAP shape: {morpho_umap.shape}")
print(f"   Morphology columns: {morpho_umap.columns.tolist()}")
print(f"   Sample Morphology data:\n{morpho_umap.head(3)}")

# ============================================================================
# STEP 2: Load cell ID mapping
# ============================================================================
print("\n[2] Loading cell ID mapping...")

cell_id_path = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/scala/exon_norm_paired_2000.csv"

# Read the gene expression file to get cell IDs in order
gene_expr_df = pd.read_csv(cell_id_path)

# Get cell IDs (assuming first column or 'cell id' column)
if 'cell id' in gene_expr_df.columns:
    cell_ids = gene_expr_df['cell id'].values
    print("   Using 'cell id' column")
elif gene_expr_df.columns[0] in ['Unnamed: 0', '']:
    gene_expr_df = pd.read_csv(cell_id_path, index_col=0)
    cell_ids = gene_expr_df.index.values
    print("   Using index as cell IDs")
else:
    # First column is cell ID
    cell_ids = gene_expr_df.iloc[:, 0].values
    print("   Using first column as cell IDs")

print(f"   Total cell IDs: {len(cell_ids)}")
print(f"   First 5 cell IDs: {cell_ids[:5].tolist()}")

# Verify the number of cells matches
if len(cell_ids) != len(gex_umap):
    print(f"\n   WARNING: Cell ID count ({len(cell_ids)}) != GEX UMAP count ({len(gex_umap)})")
    print(f"   Using minimum of both: {min(len(cell_ids), len(gex_umap))}")
    n_cells = min(len(cell_ids), len(gex_umap), len(morpho_umap))
    cell_ids = cell_ids[:n_cells]
    gex_umap = gex_umap.iloc[:n_cells]
    morpho_umap = morpho_umap.iloc[:n_cells]
else:
    n_cells = len(cell_ids)

print(f"   Final number of cells: {n_cells}")

# ============================================================================
# STEP 3: Create Sample_ID to barcode mapping
# ============================================================================
print("\n[3] Creating Sample_ID to barcode mapping...")

# Create mapping: Sample_ID (index) -> cell_id (barcode)
sample_id_to_barcode = {i: cell_ids[i] for i in range(n_cells)}

print(f"   Mapping created for {len(sample_id_to_barcode)} cells")
print(f"   Example mappings:")
for i in range(min(3, n_cells)):
    print(f"     Sample_ID {i} -> {sample_id_to_barcode[i]}")

# ============================================================================
# STEP 4: Extract UMAP coordinates
# ============================================================================
print("\n[4] Extracting UMAP coordinates...")

# GEX coordinates
gex_coords = gex_umap[['Coord_1', 'Coord_2']].values
print(f"   GEX coordinates shape: {gex_coords.shape}")
print(f"   GEX coordinate range: X=[{gex_coords[:, 0].min():.2f}, {gex_coords[:, 0].max():.2f}], Y=[{gex_coords[:, 1].min():.2f}, {gex_coords[:, 1].max():.2f}]")

# Morphology coordinates
morpho_coords = morpho_umap[['Coord_1', 'Coord_2']].values
print(f"   Morphology coordinates shape: {morpho_coords.shape}")
print(f"   Morphology coordinate range: X=[{morpho_coords[:, 0].min():.2f}, {morpho_coords[:, 0].max():.2f}], Y=[{morpho_coords[:, 1].min():.2f}, {morpho_coords[:, 1].max():.2f}]")

# ============================================================================
# STEP 5: Find nearest neighbors - GEX to Morphology
# ============================================================================
print("\n[5] Finding nearest neighbors: GEX -> Morphology...")

# Build nearest neighbor model on Morphology data
nn_morpho = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
nn_morpho.fit(morpho_coords)

# For each GEX point, find nearest Morphology point
distances_gex_to_morpho, indices_gex_to_morpho = nn_morpho.kneighbors(gex_coords)

# Create results DataFrame
results_gex_to_morpho = []
for i in range(n_cells):
    gex_sample_id = gex_umap.loc[i, 'Sample_ID']
    gex_barcode = sample_id_to_barcode[gex_sample_id]
    
    morpho_sample_id = morpho_umap.loc[indices_gex_to_morpho[i][0], 'Sample_ID']
    morpho_barcode = sample_id_to_barcode[morpho_sample_id]
    
    distance = distances_gex_to_morpho[i][0]
    
    results_gex_to_morpho.append({
        'GEX_barcode': gex_barcode,
        'Morphology_barcode': morpho_barcode,
        'distance': distance
    })

df_gex_to_morpho = pd.DataFrame(results_gex_to_morpho)

print(f"   ✓ Found {len(df_gex_to_morpho)} nearest neighbor pairs")
print(f"   Average distance: {df_gex_to_morpho['distance'].mean():.4f}")
print(f"   Min distance: {df_gex_to_morpho['distance'].min():.4f}")
print(f"   Max distance: {df_gex_to_morpho['distance'].max():.4f}")

print(f"\n   Sample results:")
print(df_gex_to_morpho.head(5))

# ============================================================================
# STEP 6: Find nearest neighbors - Morphology to GEX
# ============================================================================
print("\n[6] Finding nearest neighbors: Morphology -> GEX...")

# Build nearest neighbor model on GEX data
nn_gex = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
nn_gex.fit(gex_coords)

# For each Morphology point, find nearest GEX point
distances_morpho_to_gex, indices_morpho_to_gex = nn_gex.kneighbors(morpho_coords)

# Create results DataFrame
results_morpho_to_gex = []
for i in range(n_cells):
    morpho_sample_id = morpho_umap.loc[i, 'Sample_ID']
    morpho_barcode = sample_id_to_barcode[morpho_sample_id]
    
    gex_sample_id = gex_umap.loc[indices_morpho_to_gex[i][0], 'Sample_ID']
    gex_barcode = sample_id_to_barcode[gex_sample_id]
    
    distance = distances_morpho_to_gex[i][0]
    
    results_morpho_to_gex.append({
        'Morphology_barcode': morpho_barcode,
        'GEX_barcode': gex_barcode,
        'distance': distance
    })

df_morpho_to_gex = pd.DataFrame(results_morpho_to_gex)

print(f"   ✓ Found {len(df_morpho_to_gex)} nearest neighbor pairs")
print(f"   Average distance: {df_morpho_to_gex['distance'].mean():.4f}")
print(f"   Min distance: {df_morpho_to_gex['distance'].min():.4f}")
print(f"   Max distance: {df_morpho_to_gex['distance'].max():.4f}")

print(f"\n   Sample results:")
print(df_morpho_to_gex.head(5))

# ============================================================================
# STEP 7: Save results
# ============================================================================
print("\n[7] Saving results...")

output_dir = "/home/users/turbodu/kzlinlab/projects/morpho_integration/out/turbo/writeup23/NN"

# Save GEX -> Morphology (with distance)
output_file_1 = os.path.join(output_dir, "nearest_neighbors_GEX_to_Morphology.csv")
df_gex_to_morpho.to_csv(output_file_1, index=False)
print(f"   ✓ Saved GEX -> Morphology: {output_file_1}")

# Save GEX -> Morphology (without distance, only barcodes)
output_file_1_simple = os.path.join(output_dir, "nearest_neighbors_GEX_to_Morphology_barcodes_only.csv")
df_gex_to_morpho[['GEX_barcode', 'Morphology_barcode']].to_csv(output_file_1_simple, index=False)
print(f"   ✓ Saved GEX -> Morphology (barcodes only): {output_file_1_simple}")

# Save Morphology -> GEX (with distance)
output_file_2 = os.path.join(output_dir, "nearest_neighbors_Morphology_to_GEX.csv")
df_morpho_to_gex.to_csv(output_file_2, index=False)
print(f"   ✓ Saved Morphology -> GEX: {output_file_2}")

# Save Morphology -> GEX (without distance, only barcodes)
output_file_2_simple = os.path.join(output_dir, "nearest_neighbors_Morphology_to_GEX_barcodes_only.csv")
df_morpho_to_gex[['Morphology_barcode', 'GEX_barcode']].to_csv(output_file_2_simple, index=False)
print(f"   ✓ Saved Morphology -> GEX (barcodes only): {output_file_2_simple}")

# ============================================================================
# STEP 8: Summary statistics
# ============================================================================
print("\n[8] Summary statistics...")

# Check for reciprocal nearest neighbors
print("\n   Checking for reciprocal nearest neighbors...")
reciprocal_pairs = 0
for i in range(n_cells):
    gex_barcode = df_gex_to_morpho.loc[i, 'GEX_barcode']
    morpho_nn_of_gex = df_gex_to_morpho.loc[i, 'Morphology_barcode']
    
    # Find this morphology cell in the reverse mapping
    morpho_row = df_morpho_to_gex[df_morpho_to_gex['Morphology_barcode'] == morpho_nn_of_gex]
    if len(morpho_row) > 0:
        gex_nn_of_morpho = morpho_row.iloc[0]['GEX_barcode']
        if gex_nn_of_morpho == gex_barcode:
            reciprocal_pairs += 1

print(f"   Reciprocal nearest neighbor pairs: {reciprocal_pairs} / {n_cells} ({reciprocal_pairs/n_cells*100:.2f}%)")

# Distance distribution
print("\n   Distance distribution (GEX -> Morphology):")
print(f"     Mean: {df_gex_to_morpho['distance'].mean():.4f}")
print(f"     Median: {df_gex_to_morpho['distance'].median():.4f}")
print(f"     Std: {df_gex_to_morpho['distance'].std():.4f}")
print(f"     25th percentile: {df_gex_to_morpho['distance'].quantile(0.25):.4f}")
print(f"     75th percentile: {df_gex_to_morpho['distance'].quantile(0.75):.4f}")

print("\n   Distance distribution (Morphology -> GEX):")
print(f"     Mean: {df_morpho_to_gex['distance'].mean():.4f}")
print(f"     Median: {df_morpho_to_gex['distance'].median():.4f}")
print(f"     Std: {df_morpho_to_gex['distance'].std():.4f}")
print(f"     25th percentile: {df_morpho_to_gex['distance'].quantile(0.25):.4f}")
print(f"     75th percentile: {df_morpho_to_gex['distance'].quantile(0.75):.4f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("NEAREST NEIGHBOR ANALYSIS COMPLETE")
print("=" * 80)

print("\n📊 Summary:")
print(f"   Total cells analyzed: {n_cells}")
print(f"   Reciprocal pairs: {reciprocal_pairs} ({reciprocal_pairs/n_cells*100:.2f}%)")

print("\n📁 Output files:")
print(f"   1. {output_file_1}")
print(f"      - Columns: GEX_barcode, Morphology_barcode, distance")
print(f"      - Each GEX cell's nearest Morphology neighbor")
print(f"\n   2. {output_file_1_simple}")
print(f"      - Columns: GEX_barcode, Morphology_barcode")
print(f"      - Same as #1 but without distance")
print(f"\n   3. {output_file_2}")
print(f"      - Columns: Morphology_barcode, GEX_barcode, distance")
print(f"      - Each Morphology cell's nearest GEX neighbor")
print(f"\n   4. {output_file_2_simple}")
print(f"      - Columns: Morphology_barcode, GEX_barcode")
print(f"      - Same as #3 but without distance")

print("\n" + "=" * 80)
print("✅ Analysis completed successfully!")
print("=" * 80)
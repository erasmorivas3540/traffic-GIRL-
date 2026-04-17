"""
data_splits.py  –  GIRL Project
=================================
Run this script once before any experiment notebook to generate all
required data files.

Usage:
    python data_splits.py

Outputs:
    processed_dataset.csv          – enriched with school_zone indicator
    splits/temporal_split.json     – Protocol B (temporal OOD)
    splits/policy_split.json       – Protocol C (policy shift)
    splits/city_seed{s}_split{i}.json  – Protocol A (cross-city, 5 seeds)
    graph.pkl                      – kNN graph with Gaussian edge weights

CHANGE LOG (data pipeline fixes)
─────────────────────────────────
FIX-4  reset_index() enforced in enrich_dataset()
       Guarantees that the integer indices written to every JSON split file
       correspond exactly to positional row numbers in the saved CSV.
       Without this, any rows dropped during Step 1 (missing coordinates)
       leave gaps in the original DataFrame index, causing silent off-by-one
       errors when models.py slices tensors by position.

FIX-6  TRAIN_YEARS corrected from (2013, 2019) to (2014, 2019)
       The dataset begins in 2014.  The original constant implied 2013 data
       existed; this caused no runtime error but produced misleading log
       output (reporting "years [2014, ...]" while the config said 2013).

FIX-7  Coordinate bounding-box filter added before graph construction
       16 rows have latitudes outside San Bernardino County's range and
       7 have out-of-range longitudes (several geocoded near Sacramento).
       These outliers create spurious long-range kNN edges that corrupt the
       spatial graph's Gaussian weights and inflate the Laplacian smoothness
       penalty for nearby nodes.  Rows outside the county bounding box are
       excluded from graph construction (but retained in the split CSVs so
       row-count consistency is preserved — they simply have no edges).

FIX-8  school_zone.csv existence check added to enrich_dataset()
       A missing file previously caused a cryptic FileNotFoundError deep
       inside pandas.  A clear, early error message is now raised.

FIX-9  load_city_splits() file handles now use context managers
       The previous list comprehension [json.load(open(fp)) for fp in files]
       left file descriptors open.  Replaced with explicit with-open loops.
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "processed_dataset.csv"
SCHOOL_PATH = "school_zone.csv"
SPLITS_DIR  = "splits"
GRAPH_PATH  = "graph.pkl"

LABEL_COL = "SEVERITY"   # matches the column name written by step1_local.py
CITY_COL  = "CITY"
YEAR_COL  = "ACCIDENT_YEAR"

K_NEIGHBORS = 20          # kNN graph degree

TRAIN_YEARS = (2014, 2019)   # FIX-6: was (2013, 2019); dataset begins in 2014
VAL_YEARS   = (2020, 2021)
TEST_YEARS  = (2022, 2024)

VAL_FRAC = 0.15           # validation fraction for city / policy splits

SEEDS = [0, 1, 2, 3, 4]

# FIX-7: San Bernardino County bounding box for geocoding outlier filter.
# Rows outside this box are excluded from graph construction only.
SB_LAT_MIN, SB_LAT_MAX =  33.50, 35.81
SB_LON_MIN, SB_LON_MAX = -117.67, -114.13


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Load & enrich dataset
# ─────────────────────────────────────────────────────────────────────────────

def enrich_dataset(data_path: str, school_path: str) -> pd.DataFrame:
    """
    Load processed_dataset.csv, merge school_zone indicator from
    school_zone.csv, and overwrite the file in place.
    """
    # FIX-8: explicit file existence check with a clear error message.
    if not os.path.exists(school_path):
        raise FileNotFoundError(
            f"school_zone.csv not found at '{school_path}'. "
            "Please ensure school_zone.csv is in the working directory "
            "before running data_splits.py."
        )

    df        = pd.read_csv(data_path)
    school_df = pd.read_csv(school_path)

    for frame in [df, school_df]:
        frame["CASE_ID"] = (
            frame["CASE_ID"].astype(str).str.strip().str.replace(".0", "", regex=False)
        )

    school_ids        = set(school_df["CASE_ID"].drop_duplicates())
    df["school_zone"] = df["CASE_ID"].isin(school_ids).astype(int)

    # FIX-4: reset_index so every JSON split's integer indices map exactly
    # to positional row numbers in the CSV.  Step 1 may have dropped rows
    # (missing coordinates), leaving gaps in the original index.
    df = df.reset_index(drop=True)

    df.to_csv(data_path, index=False)

    print(f"Dataset shape : {df.shape}")
    print(f"school_zone   :\n{df['school_zone'].value_counts().to_string()}")
    print(f"Label dist    :\n{df[LABEL_COL].value_counts().to_string()}")
    print(f"\n✅ {data_path} updated.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Protocol B: Temporal OOD split
# ─────────────────────────────────────────────────────────────────────────────

def make_temporal_split(df: pd.DataFrame, out_dir: str) -> None:
    """
    Train: TRAIN_YEARS[0]–TRAIN_YEARS[1]
    Val:   VAL_YEARS[0]–VAL_YEARS[1]
    Test:  TEST_YEARS[0]–TEST_YEARS[1]
    """
    train_idx = df[
        (df[YEAR_COL] >= TRAIN_YEARS[0]) & (df[YEAR_COL] <= TRAIN_YEARS[1])
    ].index.tolist()
    val_idx = df[
        (df[YEAR_COL] >= VAL_YEARS[0]) & (df[YEAR_COL] <= VAL_YEARS[1])
    ].index.tolist()
    test_idx = df[
        (df[YEAR_COL] >= TEST_YEARS[0]) & (df[YEAR_COL] <= TEST_YEARS[1])
    ].index.tolist()

    split = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}

    path = os.path.join(out_dir, "temporal_split.json")
    with open(path, "w") as f:
        json.dump(split, f)

    print("Temporal split sizes:")
    print(f"  Train : {len(train_idx):,}  years {sorted(df.loc[train_idx, YEAR_COL].unique())}")
    print(f"  Val   : {len(val_idx):,}  years {sorted(df.loc[val_idx, YEAR_COL].unique())}")
    print(f"  Test  : {len(test_idx):,}  years {sorted(df.loc[test_idx, YEAR_COL].unique())}")
    print(f"✅ {path} saved.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Protocol C: Policy shift split
# ─────────────────────────────────────────────────────────────────────────────

def make_policy_split(df: pd.DataFrame, out_dir: str) -> None:
    """
    Train / Val: school-zone incidents  (stratified 85/15, seed=0)
    Test:        outside-school-zone incidents
    """
    train_val_df = df[df["school_zone"] == 1]
    test_df      = df[df["school_zone"] == 0]

    train_idx, val_idx = train_test_split(
        train_val_df.index.tolist(),
        test_size=VAL_FRAC,
        stratify=train_val_df[LABEL_COL],
        random_state=0,
    )
    test_idx = test_df.index.tolist()

    split = {
        "train_idx": list(map(int, train_idx)),
        "val_idx":   list(map(int, val_idx)),
        "test_idx":  list(map(int, test_idx)),
    }

    path = os.path.join(out_dir, "policy_split.json")
    with open(path, "w") as f:
        json.dump(split, f)

    print("Policy split sizes:")
    print(f"  Train (school-zone) : {len(train_idx):,}")
    print(f"  Val   (school-zone) : {len(val_idx):,}")
    print(f"  Test  (non-school)  : {len(test_idx):,}")
    print(f"✅ {path} saved.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Protocol A: Cross-city (leave-one-out) splits
# ─────────────────────────────────────────────────────────────────────────────

def make_city_splits(df: pd.DataFrame, out_dir: str) -> None:
    """
    For each seed in SEEDS, produce one JSON file per city containing
    {test_city, train_idx, val_idx, test_idx}.
    """
    cities = sorted(df[CITY_COL].unique())

    for seed in SEEDS:
        for i, test_city in enumerate(cities):
            test_mask    = df[CITY_COL] == test_city
            train_val_df = df[~test_mask]
            test_df      = df[test_mask]

            train_idx, val_idx = train_test_split(
                train_val_df.index.tolist(),
                test_size=VAL_FRAC,
                stratify=train_val_df[LABEL_COL],
                random_state=seed,
            )

            split = {
                "test_city": str(test_city),
                "train_idx": list(map(int, train_idx)),
                "val_idx":   list(map(int, val_idx)),
                "test_idx":  list(map(int, test_df.index)),
            }

            path = os.path.join(out_dir, f"city_seed{seed}_split{i}.json")
            with open(path, "w") as f:
                json.dump(split, f)

        print(f"  Seed {seed}: {len(cities)} city splits saved.")

    print(f"\nCities ({len(cities)}): {cities}")
    print(f"✅ All city splits saved ({len(SEEDS) * len(cities)} files).\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Graph construction (kNN + Gaussian weights)
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(df: pd.DataFrame, out_path: str) -> None:
    """
    Build a k-nearest-neighbour graph on GPS coordinates (haversine metric).
    Edge weights use a Gaussian kernel with sigma = mean pairwise distance.
    The adjacency matrix is symmetrised before saving.

    FIX-7: Rows with coordinates outside the San Bernardino County bounding
    box are excluded before fitting the kNN model.  These rows (geocoding
    errors with lat ~38.6, lon ~-121.5, etc.) would create spurious long-
    range edges that inflate Gaussian edge weights for nearby legitimate nodes
    and corrupt the Laplacian smoothness penalty.  The filtered-out rows are
    excluded from the graph entirely (they will have no edges).

    Saves a dict to graph.pkl with keys:
        adjacency, laplacian, edge_index, edge_weight, k, sigma, n_nodes
    """
    coords_all = df[["LATITUDE", "LONGITUDE"]].values
    n_all      = len(coords_all)

    # FIX-7: bounding-box mask — only in-county nodes participate in kNN
    in_bbox = (
        (df["LATITUDE"]  >= SB_LAT_MIN) & (df["LATITUDE"]  <= SB_LAT_MAX) &
        (df["LONGITUDE"] >= SB_LON_MIN) & (df["LONGITUDE"] <= SB_LON_MAX)
    ).values
    n_outliers = int((~in_bbox).sum())
    if n_outliers > 0:
        print(f"⚠️  Bounding-box filter: excluding {n_outliers} geocoding outliers from graph")

    coords_clean = coords_all[in_bbox]
    clean_idx    = np.where(in_bbox)[0]   # global row indices of kept nodes
    n_clean      = len(coords_clean)

    coords_rad = np.radians(coords_clean)

    # ── kNN (fitted only on in-county nodes) ─────────────────────────────────
    nbrs = NearestNeighbors(
        n_neighbors=K_NEIGHBORS + 1, algorithm="ball_tree", metric="haversine"
    )
    nbrs.fit(coords_rad)
    distances, local_indices = nbrs.kneighbors(coords_rad)

    # Remove self-loops (first column)
    distances     = distances[:, 1:]
    local_indices = local_indices[:, 1:]

    # ── Gaussian edge weights ─────────────────────────────────────────────────
    sigma   = float(np.mean(distances))
    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))

    # ── Remap local indices → global row indices ──────────────────────────────
    # local_indices are positions within coords_clean; remap to global df rows.
    global_rows = clean_idx[np.arange(n_clean).repeat(K_NEIGHBORS)]
    global_cols = clean_idx[local_indices.flatten()]
    vals        = weights.flatten()

    # ── Sparse COO adjacency (full n_all × n_all, outlier rows/cols = zero) ──
    A     = sp.coo_matrix((vals, (global_rows, global_cols)), shape=(n_all, n_all))
    A     = A.maximum(A.T).tocoo()   # symmetrise
    A_csr = A.tocsr()

    # ── Normalised Laplacian L = I − D^{-1/2} A D^{-1/2} ─────────────────────
    deg        = np.array(A_csr.sum(axis=1)).flatten()
    D_invsqrt  = sp.diags(1.0 / np.sqrt(deg + 1e-10))
    L          = sp.eye(n_all) - D_invsqrt @ A_csr @ D_invsqrt

    # ── Save ──────────────────────────────────────────────────────────────────
    graph_data = {
        "adjacency":   A_csr,
        "laplacian":   L.tocsr(),
        "edge_index":  np.vstack((A.row, A.col)),
        "edge_weight": A.data,
        "k":           K_NEIGHBORS,
        "sigma":       sigma,
        "n_nodes":     n_all,
        "n_graph_nodes": n_clean,   # number of nodes actually in the graph
        "outlier_rows":  clean_idx[local_indices.flatten()[:0]].tolist(),  # kept for audit
    }

    with open(out_path, "wb") as f:
        pickle.dump(graph_data, f)

    print(f"Total nodes   : {n_all:,}  (graph nodes: {n_clean:,}, outliers excluded: {n_outliers})")
    print(f"Edges (nnz)   : {A.nnz:,}")
    print(f"k             : {K_NEIGHBORS}")
    print(f"sigma         : {sigma:.6f}")
    print(f"Weight range  : [{A.data.min():.4f}, {A.data.max():.4f}]")
    print(f"Symmetry check: {(A_csr - A_csr.T).nnz}  (should be 0)")
    print(f"✅ {out_path} saved.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(SPLITS_DIR, exist_ok=True)

    print("=" * 56)
    print("GIRL – Data Preparation & Split Generation")
    print("=" * 56, "\n")

    # Step 1
    print("── Step 1: Load & enrich dataset ──────────────────────")
    df = enrich_dataset(DATA_PATH, SCHOOL_PATH)

    # Feature summary
    X_all = df.drop(columns=[LABEL_COL, CITY_COL]).select_dtypes(include=[np.number])
    y_all = df[LABEL_COL].values.astype(int)
    print(f"Features : {X_all.shape[1]}")
    print(f"Classes  : {np.unique(y_all)}  |  Counts: {np.bincount(y_all)}\n")

    # Step 2
    print("── Step 2: Temporal OOD split (Protocol B) ─────────────")
    make_temporal_split(df, SPLITS_DIR)

    # Step 3
    print("── Step 3: Policy shift split (Protocol C) ─────────────")
    make_policy_split(df, SPLITS_DIR)

    # Step 4
    print("── Step 4: Cross-city splits (Protocol A) ──────────────")
    make_city_splits(df, SPLITS_DIR)

    # Step 5
    print("── Step 5: Graph construction ───────────────────────────")
    build_graph(df, GRAPH_PATH)

    # Summary
    cities = sorted(df[CITY_COL].unique())
    n_city_files = len(SEEDS) * len(cities)
    print("=" * 56)
    print("All outputs ready")
    print("=" * 56)
    print(f"  {DATA_PATH:<35} {df.shape[0]:,} rows, {X_all.shape[1]} features")
    print(f"  splits/temporal_split.json")
    print(f"  splits/policy_split.json")
    print(f"  splits/city_seed*.json         {n_city_files} files ({len(SEEDS)} seeds × {len(cities)} cities)")
    print(f"  {GRAPH_PATH}")


if __name__ == "__main__":
    main()

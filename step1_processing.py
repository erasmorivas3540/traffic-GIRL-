"""
step1_local.py  –  GIRL Project
================================
Local Jupyter / JupyterLab version of step1_processing_colab_EN_fixed.py.

Colab-specific code removed:
  - 'from google.colab import files'  → replaced with pd.read_csv() direct reads
  - 'files.download(output_path)'     → removed; file is saved locally automatically

USAGE
─────
1. Set the two file paths below (MAIN_PATH and VICTIM_PATH) to point at your
   raw TIMS CSV files.
2. Run the script: %run step1_local.py   OR paste into notebook cells.
3. Output: processed_dataset.csv saved in the same directory as this script
   (or wherever your notebook's working directory is).

CHANGE LOG (data pipeline fixes vs original step1)
───────────────────────────────────────────────────
FIX-1  Binary flags (BICYCLE_ACCIDENT, PEDESTRIAN_ACCIDENT, NOT_PRIVATE_PROPERTY)
       NaN means 'N' — converted to 0/1 flags, not mode-imputed.

FIX-2  Unencoded string columns (DIRECTION, CHP_VEHTYPE_AT_FAULT)
       Previously silently dropped; now encoded.

FIX-3  Near-zero variance features removed (var < 0.001)
       Reduces noise in IRM gradient penalty.

FIX-4  reset_index() after coordinate filter
       Guarantees positional alignment with JSON split files.

FIX-5  StandardScaler removed
       Scaling delegated entirely to models.py (per-fold, train-only).
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# ✏️  SET YOUR FILE PATHS HERE
# ─────────────────────────────────────────────
MAIN_PATH   = "step1_1_1_2.csv"   # main collision table
VICTIM_PATH = "step1_3.csv"        # victim / severity table
OUTPUT_PATH = "processed_dataset.csv"
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 0. Load Raw Data
# ─────────────────────────────────────────────
print("=" * 60)
print("GIRL Step 1: Data Processing  (local version)")
print("=" * 60)

for p in [MAIN_PATH, VICTIM_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Required input file not found: '{p}'\n"
            "Update MAIN_PATH / VICTIM_PATH at the top of this script."
        )

df_main   = pd.read_csv(MAIN_PATH,   low_memory=False)
df_victim = pd.read_csv(VICTIM_PATH, low_memory=False)

print(f"\n[0] Loaded main table:   {df_main.shape}")
print(f"[0] Loaded victim table: {df_victim.shape}")

# ─────────────────────────────────────────────
# 1. Extract Label from Victim Table
#    Each CASE_ID may have multiple victims.
#    We take the maximum severity (most severe victim) as the accident label.
# ─────────────────────────────────────────────
label_df = (
    df_victim
    .groupby("CASE_ID")["severity_1_4"]
    .max()
    .reset_index()
    .rename(columns={"severity_1_4": "SEVERITY"})
)

print(f"\n[1] Extracted labels (one per CASE_ID): {label_df.shape}")
print("    Label distribution:")
print(label_df["SEVERITY"].value_counts().sort_index().to_string())

# ─────────────────────────────────────────────
# 2. Merge Main Table with Labels
# ─────────────────────────────────────────────
df = df_main.merge(label_df, on="CASE_ID", how="inner")
print(f"\n[2] After merge: {df.shape}")

# ─────────────────────────────────────────────
# 2b. Binarise severity label
#     Raw SEVERITY 1-4 ordinal scale:
#       1 = complaint of pain (minor)
#       2 = visible injury
#       3 = severe injury      ← positive class
#       4 = fatal              ← positive class
#     Binary label: 1 if fatal or severe (SEVERITY >= 3), else 0.
#     Matches paper definition: "positive class denotes fatal or
#     serious-injury crashes" (~23% of incidents).
# ─────────────────────────────────────────────
df["SEVERITY"] = (df["SEVERITY"] >= 3).astype(int)
print(f"\n[2b] Binary label created (1 = severe/fatal, 0 = minor/visible):")
print(f"     {df['SEVERITY'].value_counts().sort_index().to_dict()}")
print(f"     Positive rate: {df['SEVERITY'].mean()*100:.1f}%")

# ─────────────────────────────────────────────
# 3. Spatial Coordinate Processing
# ─────────────────────────────────────────────
df["LATITUDE"]  = df["LATITUDE"].fillna(df["POINT_Y"])
df["LONGITUDE"] = df["LONGITUDE"].fillna(df["POINT_X"])

before = len(df)
df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
print(f"\n[3] Spatial processing: dropped {before - len(df)} rows without coordinates, kept {len(df)}")

df = df.drop(columns=["POINT_X", "POINT_Y"])

# FIX-4: reset_index so positional row numbers match JSON split indices exactly.
df = df.reset_index(drop=True)

# ─────────────────────────────────────────────
# 4. Drop High-Missing and Unnecessary Columns
# ─────────────────────────────────────────────
miss_rate      = df.isnull().mean()
drop_high_miss = miss_rate[miss_rate > 0.7].index.tolist()

drop_manual = [
    "COLLISION_DATE",
    "PRIMARY_RD", "SECONDARY_RD",
    "PRIMARY_RAMP", "SECONDARY_RAMP",
    "PCF_CODE_OF_VIOL", "PCF_VIOL_SUBSECTION",
    "COLLISION_SEVERITY",
    "NUMBER_KILLED", "NUMBER_INJURED",
    "COUNT_SEVERE_INJ", "COUNT_VISIBLE_INJ",
    "COUNT_COMPLAINT_PAIN", "COUNT_PED_KILLED",
    "COUNT_PED_INJURED", "COUNT_BICYCLIST_KILLED",
    "COUNT_BICYCLIST_INJURED", "COUNT_MC_KILLED", "COUNT_MC_INJURED",
]

drop_cols = list(set(drop_high_miss + drop_manual))
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

print(f"\n[4] Dropped {len(drop_cols)} columns, remaining columns: {df.shape[1]}")

# ─────────────────────────────────────────────
# 4b. FIX-1  Binary flag columns
#     BICYCLE_ACCIDENT, PEDESTRIAN_ACCIDENT: only 'Y' appears when present;
#     NaN means 'N'. Imputing with mode would mislabel the majority.
#     NOT_PRIVATE_PROPERTY: all rows are 'Y' → zero variance → drop.
# ─────────────────────────────────────────────
for flag_col in ["BICYCLE_ACCIDENT", "PEDESTRIAN_ACCIDENT"]:
    if flag_col in df.columns:
        df[flag_col] = (df[flag_col] == "Y").astype(int)
        print(f"\n[4b] {flag_col}: converted to binary flag (1=Y, 0=NaN/other)")
        print(f"     Value counts: {df[flag_col].value_counts().to_dict()}")

if "NOT_PRIVATE_PROPERTY" in df.columns:
    df = df.drop(columns=["NOT_PRIVATE_PROPERTY"])
    print("\n[4b] NOT_PRIVATE_PROPERTY: dropped (all-Y, zero variance)")

# ─────────────────────────────────────────────
# 4c. FIX-2  CHP_VEHTYPE_AT_FAULT grouping
#     40+ rare codes grouped into 8 semantic buckets before encoding.
# ─────────────────────────────────────────────
VEHTYPE_MAP = {
    "01": "passenger_car",   # passenger car
    "02": "passenger_car",   # station wagon
    "04": "motorcycle",      # motorcycle / scooter
    "07": "truck",           # pickup or panel truck
    "08": "truck",           # truck or truck tractor
    "22": "pedestrian",      # pedestrian
    "25": "bicycle",         # bicycle
    "60": "pedestrian",      # pedestrian (alt code)
    "91": "emergency",       # emergency vehicle
    "99": "other",           # other / unknown
    "- ": "unknown",         # blank / not stated
}
if "CHP_VEHTYPE_AT_FAULT" in df.columns:
    df["CHP_VEHTYPE_AT_FAULT"] = (
        df["CHP_VEHTYPE_AT_FAULT"]
        .fillna("unknown")
        .map(lambda x: VEHTYPE_MAP.get(str(x).strip(), "other"))
    )
    print(f"\n[4c] CHP_VEHTYPE_AT_FAULT: grouped into buckets")
    print(f"     {df['CHP_VEHTYPE_AT_FAULT'].value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 5. Split Feature Types
# ─────────────────────────────────────────────
meta_cols = ["CASE_ID", "CITY", "COUNTY", "ACCIDENT_YEAR", "LATITUDE", "LONGITUDE"]

cat_encode_cols = [
    "DAY_OF_WEEK", "CHP_SHIFT", "WEATHER_1", "WEATHER_2",
    "INTERSECTION", "STATE_HWY_IND", "LOCATION_TYPE",
    "TOW_AWAY", "PRIMARY_COLL_FACTOR", "PCF_VIOL_CATEGORY",
    "HIT_AND_RUN", "TYPE_OF_COLLISION", "MVIW", "PED_ACTION",
    "ROAD_SURFACE", "ROAD_COND_1", "ROAD_COND_2",
    "LIGHTING", "CONTROL_DEVICE", "STWD_VEHTYPE_AT_FAULT",
    "CHP_ROAD_TYPE", "BEAT_TYPE", "POPULATION",
    # FIX-2: previously silently dropped string columns
    "DIRECTION",             # 4 compass directions + missing
    "CHP_VEHTYPE_AT_FAULT",  # grouped into 8 buckets above
]
cat_encode_cols = [c for c in cat_encode_cols if c in df.columns]

exclude  = set(meta_cols + cat_encode_cols + ["SEVERITY"])
num_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]

print("\n[5] Feature split:")
print(f"    Numeric columns  : {len(num_cols)}")
print(f"    Categorical cols : {len(cat_encode_cols)}")

# ─────────────────────────────────────────────
# 6. Handle Remaining Missing Values
# ─────────────────────────────────────────────
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_encode_cols:
    mode = df[c].mode()
    df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")

print(f"\n[6] Remaining missing values after imputation: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 7. FIX-5: StandardScaler removed
#    Raw values written to CSV; models.py owns per-fold scaling.
# ─────────────────────────────────────────────
print(f"\n[7] Numeric features left unscaled (scaling delegated to models.py per fold)")
print(f"    {len(num_cols)} numeric columns retained as-is")

# ─────────────────────────────────────────────
# 8. One-Hot Encode Categorical Features
# ─────────────────────────────────────────────
df_encoded = pd.get_dummies(df, columns=cat_encode_cols, drop_first=False, dtype=int)
print(f"\n[8] After one-hot encoding: {df_encoded.shape}")

# FIX-3: Drop near-zero variance dummies (var < 0.001).
_meta_and_label = ["CASE_ID", "CITY", "COUNTY", "ACCIDENT_YEAR",
                   "LATITUDE", "LONGITUDE", "SEVERITY"]
_feat_cols_enc  = [c for c in df_encoded.columns if c not in _meta_and_label]
_num_enc_feats  = df_encoded[_feat_cols_enc].select_dtypes(include=[np.number])
_var            = _num_enc_feats.var()
_drop_low_var   = _var[_var < 0.001].index.tolist()
if _drop_low_var:
    df_encoded = df_encoded.drop(columns=_drop_low_var)
    print(f"[8b] Dropped {len(_drop_low_var)} near-zero-variance features (var < 0.001)")
    print(f"     Remaining columns: {df_encoded.shape[1]}")

# ─────────────────────────────────────────────
# 9. Reorder Columns
# ─────────────────────────────────────────────
id_cols    = ["CASE_ID", "CITY", "COUNTY", "ACCIDENT_YEAR"]
coord_cols = ["LATITUDE", "LONGITUDE"]
label_col  = ["SEVERITY"]

feat_cols = [c for c in df_encoded.columns if c not in id_cols + coord_cols + label_col]
final_df  = df_encoded[id_cols + coord_cols + label_col + feat_cols]

# ─────────────────────────────────────────────
# 10. Dataset Statistics
# ─────────────────────────────────────────────
print("\n[9] Year distribution:")
yr = final_df["ACCIDENT_YEAR"].value_counts().sort_index()
for y, cnt in yr.items():
    tag = "TRAIN" if y <= 2019 else ("VAL" if y <= 2021 else "TEST")
    print(f"    {y}: {cnt:4d}  [{tag}]")

print("\n[10] Label distribution (SEVERITY):")
print(final_df["SEVERITY"].value_counts().sort_index().to_string())

# ─────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n{'='*60}")
print("✅ Step 1 Complete!")
print(f"   Final shape : {final_df.shape}")
print(f"   Saved to   : {os.path.abspath(OUTPUT_PATH)}")
print(f"{'='*60}")

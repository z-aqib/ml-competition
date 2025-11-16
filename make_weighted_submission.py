import os
import numpy as np
import pandas as pd

# ===== CONFIG =====
ANALYZING_DIR = "analyzing"
META_PATH = os.path.join(ANALYZING_DIR, "submissions_metadata.csv")

OUTPUT_DIR = "submissions"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "random_Zuha.csv")

ID_COL = "row_id"      # change if your id column has a different name
PRED_COL_NAME = "target"

ALPHA = 2.0            # score^ALPHA; try 1.0, 1.5, 2.0 etc.
THRESHOLD = 0.5        # decision threshold for final label


def main():
    # 1. Load metadata
    meta = pd.read_csv(META_PATH)

    # Keep only rows with a non-empty kaggle_score
    meta = meta[meta["kaggle_score"].notna() & (meta["kaggle_score"] > 0)]

    if meta.empty:
        raise ValueError("No rows with kaggle_score found in submissions_metadata.csv")

    print("Using the following submissions for ensemble:")
    for _, row in meta.iterrows():
        print(f"  {row['output_file']}  (score={row['kaggle_score']})")

    files = meta["output_file"].tolist()
    scores = meta["kaggle_score"].astype(float).values

    # 2. Build weights from scores
    weights = np.power(scores, ALPHA)
    weights = weights / weights.sum()
    print("\nNormalized weights:")
    for f, w in zip(files, weights):
        print(f"  {f}: {w:.4f}")

    # 3. Load and merge all submissions on ID_COL
    ensemble_df = None
    pred_cols = []

    for idx, (fname, w) in enumerate(zip(files, weights)):
        fpath = os.path.join(ANALYZING_DIR, fname)
        df = pd.read_csv(fpath)

        if ID_COL not in df.columns:
            raise ValueError(f"{fpath} does not contain '{ID_COL}' column")

        if PRED_COL_NAME not in df.columns:
            raise ValueError(f"{fpath} does not contain '{PRED_COL_NAME}' column")

        # Keep [row_id, target] and rename target
        col_name = f"pred_{idx}"
        df = df[[ID_COL, PRED_COL_NAME]].rename(columns={PRED_COL_NAME: col_name})
        pred_cols.append(col_name)

        if ensemble_df is None:
            ensemble_df = df
        else:
            # Inner join to keep only ids present in all submissions
            ensemble_df = ensemble_df.merge(df, on=ID_COL, how="inner")

    print(f"\nMerged shape: {ensemble_df.shape}")
    print(f"Prediction columns: {pred_cols}")

    # 4. Weighted vote per row
    ensemble_df[pred_cols] = ensemble_df[pred_cols].astype(float)

    # Convert weights to shape (1, n_models) for broadcasting
    w_vec = weights.reshape(1, -1)
    ensemble_df["weighted_score"] = (ensemble_df[pred_cols].values * w_vec).sum(axis=1)

    # Final label
    ensemble_df[PRED_COL_NAME] = (ensemble_df["weighted_score"] >= THRESHOLD).astype(int)

    # 5. Save final submission
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df = ensemble_df[[ID_COL, PRED_COL_NAME]].sort_values(ID_COL)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved final weighted submission to: {OUTPUT_PATH}")
    print(final_df.head())


if __name__ == "__main__":
    main()

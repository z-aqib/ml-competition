import os
import numpy as np
import pandas as pd

# ===== CONFIG =====
ANALYZING_DIR = "analyzing"
META_PATH = os.path.join(ANALYZING_DIR, "submissions_metadata.csv")

OUTPUT_DIR = "submissions"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "random_Zuha.csv")

ID_COL = "row_id"          # your id column
PRED_COL_NAME = "target"   # prediction column in each submission

ALPHA = 4.0                # higher -> best models get much more weight
THRESHOLD = 0.5            # 0.5 for binary decision

MIN_SCORE = 0.7           # ignore submissions below this Kaggle score
TOP_N = None                  # or None to use all above MIN_SCORE


def main():
    # 1. Load metadata
    meta = pd.read_csv(META_PATH)

    # Keep rows with a numeric kaggle_score
    meta = meta[meta["kaggle_score"].notna()]
    meta["kaggle_score"] = meta["kaggle_score"].astype(float)

    # Filter by minimum score
    meta = meta[meta["kaggle_score"] >= MIN_SCORE]

    if meta.empty:
        raise ValueError("No submissions >= MIN_SCORE; adjust MIN_SCORE or fill kaggle_score column.")

    # Sort by score descending
    meta = meta.sort_values("kaggle_score", ascending=False)

    # Keep only top N if specified
    if TOP_N is not None and len(meta) > TOP_N:
        meta = meta.head(TOP_N)

    print("Using these submissions for the ensemble:")
    for _, row in meta.iterrows():
        print(f"  {row['output_file']}  (score={row['kaggle_score']})")

    files = meta["output_file"].tolist()
    scores = meta["kaggle_score"].values

    # 2. Build weights from scores
    weights = np.power(scores, ALPHA)
    weights = weights / weights.sum()

    print("\nWeights:")
    for f, s, w in zip(files, scores, weights):
        print(f"  {f}: score={s:.4f}, weight={w:.4f}")

    # 3. Load and merge all submissions on ID_COL
    ensemble_df = None
    pred_cols = []

    for idx, fname in enumerate(files):
        fpath = os.path.join(ANALYZING_DIR, fname)
        df = pd.read_csv(fpath)

        if ID_COL not in df.columns:
            raise ValueError(f"{fpath} does not contain '{ID_COL}' column")

        if PRED_COL_NAME not in df.columns:
            raise ValueError(f"{fpath} does not contain '{PRED_COL_NAME}' column")

        col_name = f"pred_{idx}"
        df = df[[ID_COL, PRED_COL_NAME]].rename(columns={PRED_COL_NAME: col_name})
        pred_cols.append(col_name)

        if ensemble_df is None:
            ensemble_df = df
        else:
            ensemble_df = ensemble_df.merge(df, on=ID_COL, how="inner")

    print(f"\nMerged shape: {ensemble_df.shape}")
    print("Prediction columns:", pred_cols)

    # 4. Weighted vote
    ensemble_df[pred_cols] = ensemble_df[pred_cols].astype(float)
    w_vec = weights.reshape(1, -1)
    ensemble_df["weighted_score"] = (ensemble_df[pred_cols].values * w_vec).sum(axis=1)

    # Final hard label
    ensemble_df[PRED_COL_NAME] = (ensemble_df["weighted_score"] >= THRESHOLD).astype(int)

    # 5. Save submission
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df = ensemble_df[[ID_COL, PRED_COL_NAME]].sort_values(ID_COL)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved final weighted submission to: {OUTPUT_PATH}")
    print(final_df.head())


if __name__ == "__main__":
    main()

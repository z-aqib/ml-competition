import os
import pandas as pd

# ===== CONFIG =====
ANALYZING_DIR = "analyzing"
META_PATH = os.path.join(ANALYZING_DIR, "submissions_metadata.csv")

OUTPUT_DIR = "submissions"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "random_Zuha.csv")

ID_COL = "row_id"        # id column in each submission
PRED_COL_NAME = "target" # prediction column in each submission
TOP_N = 3                # number of top submissions to use


def main():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    # 1. Load metadata
    meta = pd.read_csv(META_PATH)

    # keep rows with numeric kaggle_score
    meta = meta[meta["kaggle_score"].notna()].copy()
    meta["kaggle_score"] = meta["kaggle_score"].astype(float)

    if meta.empty:
        raise ValueError("No rows with kaggle_score found in metadata.")

    # 2. Pick top N by kaggle_score (descending)
    meta = meta.sort_values("kaggle_score", ascending=False)
    top_meta = meta.head(TOP_N)

    if len(top_meta) < TOP_N:
        print(f"[WARN] Only found {len(top_meta)} submissions with kaggle_score; using those.")

    print("Using these submissions for majority vote:")
    for _, row in top_meta.iterrows():
        print(f"  {row['output_file']}  (score={row['kaggle_score']})")

    files = top_meta["output_file"].tolist()

    # 3. Load & merge the selected submissions
    ensemble_df = None
    pred_cols = []

    for idx, fname in enumerate(files):
        fpath = os.path.join(ANALYZING_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Submission file not found: {fpath}")

        df = pd.read_csv(fpath)

        if ID_COL not in df.columns:
            raise ValueError(f"{fpath} does not contain ID column '{ID_COL}'")
        if PRED_COL_NAME not in df.columns:
            raise ValueError(f"{fpath} does not contain prediction column '{PRED_COL_NAME}'")

        col_name = f"pred_{idx}"
        df = df[[ID_COL, PRED_COL_NAME]].rename(columns={PRED_COL_NAME: col_name})
        pred_cols.append(col_name)

        if ensemble_df is None:
            ensemble_df = df
        else:
            ensemble_df = ensemble_df.merge(df, on=ID_COL, how="inner")

    if ensemble_df is None:
        raise RuntimeError("No submissions loaded. Check your metadata and ANALYZING_DIR.")

    print(f"\nMerged shape: {ensemble_df.shape}")
    print("Prediction columns:", pred_cols)

    # 4. Majority vote row-wise
    ensemble_df[pred_cols] = ensemble_df[pred_cols].astype(int)

    # sum of 3 predictions -> 0,1,2,3
    ensemble_df["sum_votes"] = ensemble_df[pred_cols].sum(axis=1)

    # majority: if at least 2 say 1, output 1, else 0
    ensemble_df[PRED_COL_NAME] = (ensemble_df["sum_votes"] >= 2).astype(int)

    # 5. Save final submission
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df = ensemble_df[[ID_COL, PRED_COL_NAME]].sort_values(ID_COL)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved majority-vote submission to: {OUTPUT_PATH}")
    print(final_df.head())


if __name__ == "__main__":
    main()

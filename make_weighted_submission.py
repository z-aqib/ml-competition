import os
import numpy as np
import pandas as pd

# ===== CONFIG =====
ANALYZING_DIR = "analyzing"
META_PATH = os.path.join(ANALYZING_DIR, "submissions_metadata.csv")

OUTPUT_DIR = "submissions"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "random_Zuha.csv")

ID_COL = "row_id"          # your id column in each submission CSV
PRED_COL_NAME = "target"   # prediction column in each submission CSV

# ---- Good models (positive ensemble) ----
ALPHA = 4.0                # higher -> best good models get much more weight
MIN_SCORE = 0.70           # only models with kaggle_score >= MIN_SCORE are "good"

# ---- Bad models (penalty ensemble) ----
BAD_LIMIT = 0.4           # models with kaggle_score <= BAD_LIMIT are considered "bad"
BAD_WEIGHT = 0.9          # how strongly bad models penalize the final score

# ---- Final threshold on ensemble score ----
THRESHOLD = 0.5            # final decision threshold for binary label


def main():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    # 1. Load metadata
    meta = pd.read_csv(META_PATH)

    # Keep rows with a numeric kaggle_score
    meta = meta[meta["kaggle_score"].notna()].copy()
    meta["kaggle_score"] = meta["kaggle_score"].astype(float)

    if meta.empty:
        raise ValueError("No rows with kaggle_score found in metadata. Fill that column first.")

    # 2. Split into good and bad models
    good_meta = meta[meta["kaggle_score"] >= MIN_SCORE].copy()
    bad_meta = meta[meta["kaggle_score"] <= BAD_LIMIT].copy()

    if good_meta.empty:
        raise ValueError(
            f"No 'good' submissions found with kaggle_score >= {MIN_SCORE}. "
            f"Lower MIN_SCORE or update submissions_metadata.csv."
        )

    print("=== GOOD MODELS (used for main vote) ===")
    for _, row in good_meta.sort_values("kaggle_score", ascending=False).iterrows():
        print(f"  {row['output_file']}  (score={row['kaggle_score']})")

    if bad_meta.empty:
        print("\n=== BAD MODELS ===")
        print("  None (no submissions <= BAD_LIMIT). Penalty will be zero.")
    else:
        print("\n=== BAD MODELS (used as penalty) ===")
        for _, row in bad_meta.sort_values("kaggle_score", ascending=True).iterrows():
            print(f"  {row['output_file']}  (score={row['kaggle_score']})")

    good_files = good_meta["output_file"].tolist()
    good_scores = good_meta["kaggle_score"].values

    bad_files = bad_meta["output_file"].tolist()
    bad_scores = bad_meta["kaggle_score"].values

    # 3. Build weights for good models
    good_weights = np.power(good_scores, ALPHA)
    good_weights = good_weights / good_weights.sum()

    print("\n=== GOOD MODEL WEIGHTS (normalized) ===")
    for f, s, w in zip(good_files, good_scores, good_weights):
        print(f"  {f}: score={s:.4f}, weight={w:.4f}")

    if bad_files:
        print(f"\nBAD_WEIGHT (penalty strength) = {BAD_WEIGHT}")
    else:
        print("\nBAD_WEIGHT will effectively be ignored (no bad models).")

    # 4. Load and merge submissions
    df_final = None
    good_pred_cols = []
    bad_pred_cols = []

    # ---- Load good models ----
    for idx, fname in enumerate(good_files):
        fpath = os.path.join(ANALYZING_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Good model file not found: {fpath}")

        df = pd.read_csv(fpath)

        if ID_COL not in df.columns:
            raise ValueError(f"{fpath} does not contain ID column '{ID_COL}'")
        if PRED_COL_NAME not in df.columns:
            raise ValueError(f"{fpath} does not contain prediction column '{PRED_COL_NAME}'")

        col = f"good_{idx}"
        df = df[[ID_COL, PRED_COL_NAME]].rename(columns={PRED_COL_NAME: col})
        good_pred_cols.append(col)

        df_final = df if df_final is None else df_final.merge(df, on=ID_COL, how="inner")

    # ---- Load bad models ----
    for idx, fname in enumerate(bad_files):
        fpath = os.path.join(ANALYZING_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] Bad model file not found (skipping): {fpath}")
            continue

        df = pd.read_csv(fpath)

        if ID_COL not in df.columns or PRED_COL_NAME not in df.columns:
            print(f"[WARN] {fpath} missing '{ID_COL}' or '{PRED_COL_NAME}' (skipping).")
            continue

        col = f"bad_{idx}"
        df = df[[ID_COL, PRED_COL_NAME]].rename(columns={PRED_COL_NAME: col})
        bad_pred_cols.append(col)

        df_final = df_final.merge(df, on=ID_COL, how="inner")

    # Safety check
    if df_final is None:
        raise RuntimeError("No submissions could be loaded. Check your ANALYZING_DIR and metadata.")

    print(f"\nMerged shape: {df_final.shape}")
    print("Good prediction columns:", good_pred_cols)
    print("Bad prediction columns:", bad_pred_cols)

    # 5. Compute ensemble scores
    df_final[good_pred_cols] = df_final[good_pred_cols].astype(float)
    good_weight_vec = good_weights.reshape(1, -1)
    # main vote from good models
    good_vote = (df_final[good_pred_cols].values * good_weight_vec).sum(axis=1)

    # penalty from bad models (average)
    if bad_pred_cols:
        df_final[bad_pred_cols] = df_final[bad_pred_cols].astype(float)
        bad_vote = df_final[bad_pred_cols].mean(axis=1).values
    else:
        bad_vote = np.zeros_like(good_vote)

    # final score = good_vote - BAD_WEIGHT * bad_vote
    final_score = good_vote - BAD_WEIGHT * bad_vote

    # 6. Final hard label
    df_final[PRED_COL_NAME] = (final_score >= THRESHOLD).astype(int)

    # 7. Save submission
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df = df_final[[ID_COL, PRED_COL_NAME]].sort_values(ID_COL)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved final ensemble submission to: {OUTPUT_PATH}")
    print(final_df.head())


if __name__ == "__main__":
    main()

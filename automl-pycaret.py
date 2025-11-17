# auto_ml_pycaret.py

import pandas as pd
from pycaret.classification import *

# =======================================
# 1. Paths (change if your paths differ)
# =======================================
TRAIN_PATH = "./data/train_part1.json"   # or "./data/trainp1.json"
TEST_PATH  = "./data/test.json"
SUB_PATH   = "./auto-ml/pycaret_submission.csv"

# =======================================
# 2. Load JSON data
# =======================================
# If your JSON is line-delimited (one object per line), keep lines=True.
# If it's a normal JSON list, remove lines=True.
train_raw = pd.read_json(TRAIN_PATH, lines=True)
test_raw  = pd.read_json(TEST_PATH,  lines=True)

# train_raw columns: ["id", "label", "image_embedding", "text_embedding"]
# test_raw  columns: ["id", "image_embedding", "text_embedding"]

# =======================================
# 3. Flatten embeddings into tabular features
# =======================================
img_train = pd.DataFrame(
    train_raw["image_embedding"].tolist(),
    columns=[f"img_{i}" for i in range(512)]
)

txt_train = pd.DataFrame(
    train_raw["text_embedding"].tolist(),
    columns=[f"text_{i}" for i in range(512)]
)

# Add label
train_df = pd.concat([img_train, txt_train, train_raw["label"]], axis=1)

# For test: same feature engineering, but keep IDs separate
img_test = pd.DataFrame(
    test_raw["image_embedding"].tolist(),
    columns=[f"img_{i}" for i in range(512)]
)

txt_test = pd.DataFrame(
    test_raw["text_embedding"].tolist(),
    columns=[f"text_{i}" for i in range(512)]
)

test_features = pd.concat([img_test, txt_test], axis=1)
test_ids = test_raw["id"]  # will be used as row_id in submission

print("Train shape:", train_df.shape)
print("Test shape:", test_features.shape)

# =======================================
# 4. PyCaret setup
# =======================================
clf = setup(
    data=train_df,
    target="label",
    normalize=True,            # embeddings benefit from normalization
    fix_imbalance=True,        # just in case classes are imbalanced
    session_id=123,            # for reproducibility
    fold=5,
    fold_shuffle=True,
    use_gpu=True,              # if GPU available in Kaggle, PyCaret will use it
    silent=True,
    verbose=False,
    # explicitly optimize for macro F1 (good for this comp)
    # in newer PyCaret versions: you can also pass 'optimize' later to compare_models
)

# =======================================
# 5. Compare models & pick the best (by F1)
# =======================================
best_model = compare_models(sort="F1")  # macro F1 for binary works well

print("Best model:", best_model)

# =======================================
# 6. Finalize model on full training data
# =======================================
final_model = finalize_model(best_model)

# =======================================
# 7. Predict on test set
# =======================================
preds = predict_model(final_model, data=test_features)

# PyCaret adds columns like:
# 'prediction_label' (0/1) and 'prediction_score'
submission = pd.DataFrame({
    "row_id": test_ids,
    "target": preds["prediction_label"].astype(int)
})

# =======================================
# 8. Save submission
# =======================================
submission.to_csv(SUB_PATH, index=False)
print(f"Saved submission to: {SUB_PATH}")

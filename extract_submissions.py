import os
import subprocess
import csv
import re
from pathlib import Path

# ===== CONFIG =====
SUBMISSIONS_DIR = "submissions"   # folder where your CSVs live(d)
OUTPUT_DIR = "analyzing"          # new folder to collect all versions

os.makedirs(OUTPUT_DIR, exist_ok=True)

META_PATH = os.path.join(OUTPUT_DIR, "submissions_metadata.csv")


def run(cmd: str) -> str:
    """Run a shell command and return stdout as text."""
    return subprocess.check_output(cmd, shell=True, text=True, encoding="utf-8")


# ===== 0. Load existing metadata (if any) to avoid duplicates =====
existing_keys = set()  # (original_path, commit_hash)

if os.path.exists(META_PATH):
    print(f"Found existing metadata at {META_PATH}, loading to avoid duplicates...")
    with open(META_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # be robust if some old rows are missing fields
            original_path = row.get("original_path")
            commit_hash = row.get("commit_hash")
            if original_path and commit_hash:
                existing_keys.add((original_path, commit_hash))

    print(f"Loaded {len(existing_keys)} existing (path, commit) pairs.")
else:
    print("No existing metadata file found. Starting fresh.")


# ===== 1. Find every CSV that ever existed under submissions/ =====
print("Collecting submission file names from git history...")

log_output = run(f'git log --name-only --pretty=format: -- "{SUBMISSIONS_DIR}"')
all_paths = set()

for line in log_output.splitlines():
    line = line.strip()
    if not line:
        continue
    # only keep CSVs under submissions/
    if line.startswith(SUBMISSIONS_DIR) and line.endswith(".csv"):
        all_paths.add(line)

all_paths = sorted(all_paths)
print("Found files:")
for p in all_paths:
    print("  ", p)

# ===== 2. For each file, get all its versions from history =====
new_metadata_rows = []

for path in all_paths:
    print(f"\nProcessing file history for: {path}")
    # git log for this file only
    # %h = short hash, %cd = date, %s = commit message
    log_cmd = f'git log --pretty=format:"%h|%cd|%s" --date=short -- "{path}"'
    file_log = run(log_cmd).strip()

    if not file_log:
        continue

    for line in file_log.splitlines():
        # split into hash, date, message
        try:
            sha, date, msg = line.split("|", 2)
        except ValueError:
            # weird line, skip
            continue

        sha = sha.strip()
        date = date.strip()
        msg = msg.strip()

        key = (path, sha)
        if key in existing_keys:
            # We already have this (file, commit) in metadata, skip
            # (and do NOT touch its CSV or kaggle_score)
            continue

        # Safe file name parts
        original_name = Path(path).name
        stem = Path(path).stem

        # Sanitize commit message snippet for filename
        snippet = msg.lower()
        snippet = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in snippet)
        snippet = snippet.strip("_")
        if len(snippet) > 40:
            snippet = snippet[:40]
        if not snippet:
            snippet = "no_msg"

        out_name = f"{stem}__{sha}__{snippet}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        # Extract file contents at that commit
        show_cmd = f'git show {sha}:"{path}"'
        try:
            file_contents = run(show_cmd)
        except subprocess.CalledProcessError:
            # file might not exist at that commit (rare edge case)
            print(f"  [WARN] Could not extract {path} at {sha}")
            continue

        # Write to analyzing/ (if the file already exists, overwrite is harmless;
        # but usually this will be new)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(file_contents)

        # Try to auto-parse a Kaggle score from the commit message
        # Heuristic: first float between 0.3 and 1.0
        kaggle_score = ""
        numbers = re.findall(r"\d+\.\d+", msg)
        for num in numbers:
            try:
                val = float(num)
            except ValueError:
                continue
            if 0.3 <= val <= 1.0:
                kaggle_score = val
                break

        new_metadata_rows.append({
            "output_file": out_name,
            "original_path": path,
            "original_name": original_name,
            "commit_hash": sha,
            "date": date,
            "commit_message": msg,
            "kaggle_score": kaggle_score,  # blank if not auto-parsed
        })

# ===== 3. Append new rows to metadata CSV (without touching existing rows) =====
fieldnames = [
    "output_file",
    "original_path",
    "original_name",
    "commit_hash",
    "date",
    "commit_message",
    "kaggle_score",
]

if not new_metadata_rows:
    print("\nNo new submissions found; metadata unchanged.")
else:
    file_exists = os.path.exists(META_PATH)
    mode = "a" if file_exists else "w"

    with open(META_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_metadata_rows)

    print(f"\nAdded {len(new_metadata_rows)} new rows to '{META_PATH}'")

print(f"CSV versions are in '{OUTPUT_DIR}'")

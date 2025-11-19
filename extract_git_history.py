import subprocess
import csv
import os

# ===== CONFIG =====
OUTPUT_CSV = "readme-helpers/git_history.csv"


def run(cmd: str) -> str:
    """
    Run a shell command and return stdout as text.
    Assumes you're inside a git repo and git is installed.
    """
    return subprocess.check_output(
        cmd,
        shell=True,
        text=True,
        encoding="utf-8",
    )


def main():
    # Git log with:
    # - full hash (%H)
    # - short date (%cd)
    # - author name (%an)
    # - subject (%s)
    # and then the list of changed files with their status (--name-status)
    log_cmd = (
        'git log '
        '--date=short '
        '--name-status '
        '--pretty=format:"COMMIT|%H|%cd|%an|%s"'
    )

    print("Running git log to collect commit history...")
    output = run(log_cmd)

    rows = []
    current = None
    files_in_current = 0

    for raw_line in output.splitlines():
        line = raw_line.rstrip("\n")

        if not line.strip():
            # empty line between commits; ignore
            continue

        if line.startswith("COMMIT|"):
            # If the previous commit had no files (very rare), still record one row
            if current is not None and files_in_current == 0:
                rows.append(
                    {
                        "commit_hash": current["commit_hash"],
                        "date": current["date"],
                        "author": current["author"],
                        "subject": current["subject"],
                        "status": "",
                        "file_path": "",
                        "old_file_path": "",
                    }
                )

            # New commit header
            parts = line.split("|", 4)  # COMMIT | hash | date | author | subject
            if len(parts) < 5:
                # Fallback in case subject has weird separators (very unlikely)
                # but we at least parse first 4 safely
                tag, commit_hash, date, author = (parts + ["", "", ""])[:4]
                subject = ""
            else:
                _, commit_hash, date, author, subject = parts

            current = {
                "commit_hash": commit_hash.strip(),
                "date": date.strip(),
                "author": author.strip(),
                "subject": subject.strip(),
            }
            files_in_current = 0
        else:
            # File change line from --name-status
            # Format:
            #   M\tpath
            #   A\tpath
            #   D\tpath
            #   R100\told_path\tnew_path   (rename)
            parts = line.split("\t")
            if not parts:
                continue

            status = parts[0].strip()
            file_path = ""
            old_file_path = ""

            if len(parts) == 2:
                # A/M/D etc.
                file_path = parts[1].strip()
            elif len(parts) >= 3:
                # Rename with score: Rxxx <old> <new>
                old_file_path = parts[1].strip()
                file_path = parts[2].strip()

            if current is None:
                # Should not happen, but guard anyway
                continue

            rows.append(
                {
                    "commit_hash": current["commit_hash"],
                    "date": current["date"],
                    "author": current["author"],
                    "subject": current["subject"],
                    "status": status,
                    "file_path": file_path,
                    "old_file_path": old_file_path,
                }
            )
            files_in_current += 1

    # Handle last commit if it had no file rows (again, rare)
    if current is not None and files_in_current == 0:
        rows.append(
            {
                "commit_hash": current["commit_hash"],
                "date": current["date"],
                "author": current["author"],
                "subject": current["subject"],
                "status": "",
                "file_path": "",
                "old_file_path": "",
            }
        )

    # ===== Write CSV =====
    fieldnames = [
        "commit_hash",
        "date",
        "author",
        "subject",
        "status",
        "file_path",
        "old_file_path",
    ]

    print(f"Writing {len(rows)} rows to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    # Make sure we're inside a git repo (simple check)
    if not os.path.isdir(".git"):
        print("ERROR: This script must be run from the root of a git repository (where .git exists).")
    else:
        main()

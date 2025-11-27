"""Utility to find and delete all 'embeddings' subfolders under a given root directory. AI generated."""
# pyright: basic
import os
import shutil
import sys
from fire import Fire # type: ignore[reportMissingImports, reportMissingTypeStubs]


def find_embeddings(root):
    matches = []
    for dirpath, dirnames, _ in os.walk(root, topdown=True):
        # copy list because we'll modify dirnames if needed
        for d in list(dirnames):
            if d == "embeddings":
                matches.append(os.path.join(dirpath, d))
                # avoid descending into it
                dirnames.remove(d)
    return matches

def delete_paths(paths, dry_run=False):
    for p in paths:
        if dry_run:
            print("[DRY-RUN] Would remove:", p)
        else:
            try:
                shutil.rmtree(p)
                print("Removed:", p)
            except Exception as e:
                print("Failed to remove", p, ":", e, file=sys.stderr)

def main(
        root: str="extraction_validator/test_extend",
        dry_run: bool=False,
        yes: bool=True,
):
    """Find and delete all 'embeddings' subfolders under the given root directory.
    Args:
      root: Root directory to start searching from.
      dry_run: If True, only print the directories that would be deleted.
      yes: If True, skip confirmation prompt before deletion.
    """


    root = os.path.abspath(root)
    matches = find_embeddings(root)
    if not matches:
        print("No 'embeddings' directories found under", root)
        return

    print("Found", len(matches), "'embeddings' directories:")
    for m in matches:
        print("  ", m)

    if dry_run:
        delete_paths(matches, dry_run=True)
        return

    if not yes:
        confirm = input("Delete these directories? Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted by user.")
            return

    delete_paths(matches, dry_run=False)

if __name__ == "__main__":
    Fire(main)
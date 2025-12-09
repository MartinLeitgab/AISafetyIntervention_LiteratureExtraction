# ...existing code...
"""A quick script to randomly sample folders from a source directory and copy them to a destination directory. AI generated."""
#!/usr/bin/env python3

import random
import shutil
from pathlib import Path
from typing import Optional
from fire import Fire  # type: ignore[reportMissingImports, reportMissingTypeStubs]

def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    i = 1
    while True:
        candidate = dest.with_name(f"{dest.name}_copy{i}")
        if not candidate.exists():
            return candidate
        i += 1

def copy_random_folders(src: str = "processed_ard",
                        dst: str = "extraction_validator/test_extend",
                        n: int = 100,
                        seed: Optional[int] = None,
                        overwrite: bool = False) -> None:
    """
    Randomly sample up to `n` immediate subfolders from `src` and copy them into `dst`.

    Args:
      src: source directory containing subfolders to sample.
      dst: destination directory to copy sampled folders into.
      n: number of folders to sample (max).
      seed: optional random seed for reproducibility.
      overwrite: if True, existing destination folders with the same name will be removed.
    """
    src_p = Path(src).expanduser().resolve()
    dst_p = Path(dst).expanduser().resolve()

    if not src_p.exists() or not src_p.is_dir():
        raise SystemExit(f"Source directory not found: {src_p}")

    dst_p.mkdir(parents=True, exist_ok=True)

    subdirs = [d for d in src_p.iterdir() if d.is_dir()]
    total = len(subdirs)
    if total == 0:
        raise SystemExit(f"No subfolders found in source: {src_p}")

    if total < n:
        print(f"Warning: requested {n} folders but only {total} available. Will copy all {total}.")
        n = total

    rnd = random.Random(seed)
    sampled = rnd.sample(subdirs, n)

    for i, folder in enumerate(sampled, start=1):
        dest_folder = dst_p / folder.name
        if overwrite and dest_folder.exists():
            print(f"[{i}/{n}] Overwriting existing {dest_folder.name}")
            shutil.rmtree(dest_folder)
        else:
            dest_folder = unique_dest(dest_folder)
            print(f"[{i}/{n}] Copying {folder.name} -> {dest_folder.name}")
        shutil.copytree(folder, dest_folder)
    print("Done.")

if __name__ == "__main__":
    Fire(copy_random_folders)
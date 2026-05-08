import os
import re
import numpy as np
from fire import Fire  # type: ignore[import]


def main(folder_path: str):
    files = os.listdir(folder_path)
    pre_scores = []
    post_scores = []

    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                matches = re.findall(r'"pre_judge_score":\s*(\d+)', content)
                pre_score = None
                post_score = None
                if matches:
                    # print(f"{file}: pre {matches}")
                    pre_score = int(matches[0])
                matches = re.findall(r'"post_judge_score":\s*(\d+)', content)
                if matches:
                    post_score = int(matches[0])
                if pre_score is None or post_score is None:
                    continue
                pre_scores.append(pre_score)
                post_scores.append(post_score)
    print(
        f"Pre scores - Mean: {np.mean(pre_scores):.2f}, Std: {np.std(pre_scores):.2f}"
    )
    print(
        f"Post scores - Mean: {np.mean(post_scores):.2f}, Std: {np.std(post_scores):.2f}"
    )


if __name__ == "__main__":
    Fire(main)

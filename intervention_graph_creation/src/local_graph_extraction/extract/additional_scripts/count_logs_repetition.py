import re
from collections import Counter

with open(
    "./AISafetyIntervention_LiteratureExtraction/logs/extraction_10-08_14-00.log",
    "r",
    encoding="utf-8",
) as f:
    lines = f.readlines()

pattern = re.compile(r"Skipping already processed or failed JSONL: (.+)")

names = []
for line in lines:
    match = pattern.search(line)
    if match:
        names.append(match.group(1))

counter = Counter(names)

for name, count in counter.items():
    if count >= 2:
        print(f"{name}: {count}")

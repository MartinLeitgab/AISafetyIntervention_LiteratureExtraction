#!/usr/bin/env python3
"""Sample NON-keyword-flagged PAs for recall estimation across importance tiers."""

import pickle
import random
import csv

with open('local/stage2_clustering/stage2.2/results/graph_stage2.2_with_metrics.pkl', 'rb') as f:
    G = pickle.load(f)

def get_structural_pa_neighbors(G, node):
    pa_neighbors = []
    for neighbor in G.successors(node):
        edge_data = G.edges[node, neighbor]
        if edge_data.get('type') != 'similarity':
            node_data = G.nodes[neighbor]
            if node_data.get('concept_category') == 'problem analysis':
                pa_neighbors.append((neighbor, node_data))
    return pa_neighbors

def is_race_keyword(name):
    name_lower = name.lower()
    return 'competi' in name_lower or 'race' in name_lower

# Collect ALL risks with eigenvector centrality
all_risks = []
for n, d in G.nodes(data=True):
    if d.get('concept_category') == 'risk':
        pa_neighbors = get_structural_pa_neighbors(G, n)
        all_risks.append({
            'risk_id': n,
            'risk_name': d.get('name', n),
            'eigenvector': d.get('eigenvector_centrality', 0),
            'pa_neighbors': pa_neighbors
        })

# Sort by eigenvector centrality
all_risks.sort(key=lambda x: -x['eigenvector'])
print(f"Total risks: {len(all_risks)}")

# Extract monopath risks with their PA info
def extract_monopath(risks):
    result = []
    for r in risks:
        if len(r['pa_neighbors']) == 1:
            pa_id, pa_data = r['pa_neighbors'][0]
            pa_name = pa_data.get('name', pa_id)
            result.append({
                'risk_id': r['risk_id'],
                'risk_name': r['risk_name'],
                'eigenvector': r['eigenvector'],
                'pa_id': pa_id,
                'pa_name': pa_name,
                'pa_description': pa_data.get('description', ''),
                'is_keyword_flagged': is_race_keyword(pa_name)
            })
    return result

# Split by tiers
tier_100 = extract_monopath(all_risks[:100])
tier_500 = extract_monopath(all_risks[100:500])
tier_1000 = extract_monopath(all_risks[500:1000])
tier_rest = extract_monopath(all_risks[1000:])

print(f"\nMonopath counts by tier:")
print(f"  Top-100: {len(tier_100)} total, {sum(1 for x in tier_100 if x['is_keyword_flagged'])} flagged, {sum(1 for x in tier_100 if not x['is_keyword_flagged'])} non-flagged")
print(f"  101-500: {len(tier_500)} total, {sum(1 for x in tier_500 if x['is_keyword_flagged'])} flagged, {sum(1 for x in tier_500 if not x['is_keyword_flagged'])} non-flagged")
print(f"  501-1000: {len(tier_1000)} total, {sum(1 for x in tier_1000 if x['is_keyword_flagged'])} flagged, {sum(1 for x in tier_1000 if not x['is_keyword_flagged'])} non-flagged")
print(f"  1001+: {len(tier_rest)} total, {sum(1 for x in tier_rest if x['is_keyword_flagged'])} flagged, {sum(1 for x in tier_rest if not x['is_keyword_flagged'])} non-flagged")

# Get NON-flagged from each tier
non_flagged_100 = [x for x in tier_100 if not x['is_keyword_flagged']]
non_flagged_500 = [x for x in tier_500 if not x['is_keyword_flagged']]
non_flagged_1000 = [x for x in tier_1000 if not x['is_keyword_flagged']]
non_flagged_rest = [x for x in tier_rest if not x['is_keyword_flagged']]

# Sample strategy
random.seed(42)
sample = []

# Stratum 1: ALL non-flagged from top-100 (expected ~5)
for item in non_flagged_100:
    rank = next(i+1 for i, r in enumerate(all_risks) if r['risk_id'] == item['risk_id'])
    sample.append({**item, 'stratum': '1_top100', 'risk_rank': rank})
print(f"\nStratum 1 (top-100 non-flagged): {len(non_flagged_100)}")

# Stratum 2: Sample from 101-500 (up to 15)
n_sample_500 = min(15, len(non_flagged_500))
sampled_500 = random.sample(non_flagged_500, n_sample_500)
for item in sampled_500:
    rank = next(i+1 for i, r in enumerate(all_risks) if r['risk_id'] == item['risk_id'])
    sample.append({**item, 'stratum': '2_101-500', 'risk_rank': rank})
print(f"Stratum 2 (101-500 non-flagged, sampled): {n_sample_500}")

# Stratum 3: Sample from 501-1000 (up to 15)
n_sample_1000 = min(15, len(non_flagged_1000))
sampled_1000 = random.sample(non_flagged_1000, n_sample_1000)
for item in sampled_1000:
    rank = next(i+1 for i, r in enumerate(all_risks) if r['risk_id'] == item['risk_id'])
    sample.append({**item, 'stratum': '3_501-1000', 'risk_rank': rank})
print(f"Stratum 3 (501-1000 non-flagged, sampled): {n_sample_1000}")

# Stratum 4: Sample from 1001+ (up to 15)
n_sample_rest = min(15, len(non_flagged_rest))
sampled_rest = random.sample(non_flagged_rest, n_sample_rest)
for item in sampled_rest:
    rank = next(i+1 for i, r in enumerate(all_risks) if r['risk_id'] == item['risk_id'])
    sample.append({**item, 'stratum': '4_1001+', 'risk_rank': rank})
print(f"Stratum 4 (1001+ non-flagged, sampled): {n_sample_rest}")

print(f"\nTotal sample: {len(sample)}")

# Output CSV
output_path = 'local/stage2_clustering/stage2.2/race_framing_recall_sample.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'stratum', 'risk_rank', 'risk_id', 'risk_name', 'eigenvector',
        'pa_id', 'pa_name', 'pa_description', 'is_race_framed', 'notes'
    ])
    writer.writeheader()
    for item in sample:
        writer.writerow({
            'stratum': item['stratum'],
            'risk_rank': item['risk_rank'],
            'risk_id': item['risk_id'],
            'risk_name': item['risk_name'],
            'eigenvector': item['eigenvector'],
            'pa_id': item['pa_id'],
            'pa_name': item['pa_name'],
            'pa_description': item['pa_description'],
            'is_race_framed': '',
            'notes': ''
        })

print(f"\nOutput: {output_path}")
print("\nAnnotation guide:")
print("  is_race_framed = 1 if PA describes human race/competition TO DEVELOP AI")
print("  is_race_framed = 0 if not about AI development race dynamics")

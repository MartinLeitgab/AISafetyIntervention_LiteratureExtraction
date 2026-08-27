#!/usr/bin/env python3
"""Stratified sample of race-framed PA nodes for manual validation."""

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

def is_race_framed(name):
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

# Sort ALL risks by eigenvector centrality
all_risks.sort(key=lambda x: -x['eigenvector'])

# Take top-100 risks by importance
top_100_risks = all_risks[:100]
print(f"Top-100 risks by eigenvector")

# Filter to monopath only
top_100_monopath = []
for r in top_100_risks:
    if len(r['pa_neighbors']) == 1:
        pa_id, pa_data = r['pa_neighbors'][0]
        top_100_monopath.append({
            'risk_id': r['risk_id'],
            'risk_name': r['risk_name'],
            'eigenvector': r['eigenvector'],
            'pa_id': pa_id,
            'pa_name': pa_data.get('name', pa_id),
            'pa_description': pa_data.get('description', ''),
            'is_race_keyword': is_race_framed(pa_data.get('name', pa_id))
        })

print(f"Top-100 monopath: {len(top_100_monopath)}")

# STRATUM 1: All race-flagged PAs from top-100 monopath
stratum_1 = [r for r in top_100_monopath if r['is_race_keyword']]
print(f"Stratum 1 (top-100 monopath race-flagged): {len(stratum_1)}")

# STRATUM 2: Random race-flagged PAs from risks outside top-100
rest_monopath = []
for r in all_risks[100:]:
    if len(r['pa_neighbors']) == 1:
        pa_id, pa_data = r['pa_neighbors'][0]
        rest_monopath.append({
            'risk_id': r['risk_id'],
            'risk_name': r['risk_name'],
            'eigenvector': r['eigenvector'],
            'pa_id': pa_id,
            'pa_name': pa_data.get('name', pa_id),
            'pa_description': pa_data.get('description', ''),
            'is_race_keyword': is_race_framed(pa_data.get('name', pa_id))
        })

rest_race_flagged = [r for r in rest_monopath if r['is_race_keyword']]
random.seed(42)
stratum_2 = random.sample(rest_race_flagged, min(16, len(rest_race_flagged)))
print(f"Stratum 2 (rest, sampled 16): {len(stratum_2)}")

# Combine
sample = stratum_1 + stratum_2
print(f"Total sample: {len(sample)}")

# Output CSV
with open('local/race_framing_sample_50.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'stratum', 'risk_rank', 'risk_id', 'risk_name', 'eigenvector',
        'pa_id', 'pa_name', 'pa_description', 'is_race_framed', 'notes'
    ])
    writer.writeheader()
    for item in stratum_1:
        # Find rank in original top-100 risks
        rank = next(i+1 for i, r in enumerate(top_100_risks) if r['risk_id'] == item['risk_id'])
        writer.writerow({
            'stratum': 1, 'risk_rank': rank,
            'risk_id': item['risk_id'], 'risk_name': item['risk_name'],
            'eigenvector': item['eigenvector'],
            'pa_id': item['pa_id'], 'pa_name': item['pa_name'],
            'pa_description': item['pa_description'],
            'is_race_framed': '', 'notes': ''
        })
    for item in stratum_2:
        writer.writerow({
            'stratum': 2, 'risk_rank': '>100',
            'risk_id': item['risk_id'], 'risk_name': item['risk_name'],
            'eigenvector': item['eigenvector'],
            'pa_id': item['pa_id'], 'pa_name': item['pa_name'],
            'pa_description': item['pa_description'],
            'is_race_framed': '', 'notes': ''
        })

print(f"\nOutput: local/race_framing_sample_50.csv")

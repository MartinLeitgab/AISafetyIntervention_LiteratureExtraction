import arxiv
import json
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import os
from community import community_louvain
import re
import matplotlib.patches as mpatches

# Load spaCy model for scientific text
nlp = spacy.load("en_core_sci_sm")

# Custom stopwords to filter noise
nlp.Defaults.stop_words |= {"current", "study", "work", "et al", "etc", "paper", "research"}

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s.,-]', '', text)  # Remove special characters except basic punctuation
    return text.strip()

# ----------------------------
# STEP 1: Fetch papers from arXiv
# ----------------------------
def fetch_ai_safety_papers(query="AI safety alignment", max_results=5, json_file="ai_safety_papers.json"):
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        results = []

        for result in client.results(search):
            summary = preprocess_text(result.summary)
            results.append({
                "id": result.entry_id.split('/')[-1],
                "title": preprocess_text(result.title),
                "authors": [author.name for author in result.authors],
                "summary": summary,
                "categories": result.categories,
                "published": result.published.strftime("%Y-%m-%d")
            })

        os.makedirs(os.path.dirname(json_file) or '.', exist_ok=True)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Fetched {len(results)} papers and saved to {json_file}")
    except Exception as e:
        print(f"❌ Error fetching papers: {e}")
        raise

# ----------------------------
# STEP 2: Extract Entities and Concepts
# ----------------------------
def extract_entities_and_concepts(text):
    try:
        doc = nlp(preprocess_text(text))
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "TECH", "CONCEPT", "METHOD", "TASK"] and not any(w in nlp.Defaults.stop_words for w in ent.text.lower().split())]
        concepts = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1 and not any(w.is_stop or w.is_punct or w in nlp.Defaults.stop_words for w in chunk)][:3]
        return entities, concepts
    except Exception as e:
        print(f"❌ Error extracting entities/concepts: {e}")
        return [], []

# ----------------------------
# STEP 3: Convert to Triples with Enhanced Relationships
# ----------------------------
def convert_to_triples(json_file="ai_safety_papers.json", output_file="graph_triples.txt"):
    try:
        with open(json_file, encoding="utf-8") as f:
            papers = json.load(f)

        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as out:
            for paper in papers:
                pid = paper["title"].replace(" ", "_").replace("\n", "_").replace(":", "_").replace(",", "_")

                for author in paper["authors"]:
                    out.write(f"{author}\tauthored_by\t{pid}\n")
                for cat in paper["categories"]:
                    out.write(f"{pid}\tbelongs_to_category\t{cat}\n")
                out.write(f"{pid}\tpublished_on\t{paper['published']}\n")

                entities, concepts = extract_entities_and_concepts(paper["summary"])
                for entity, entity_type in entities:
                    entity_clean = entity.replace(" ", "_").replace("\n", "_")
                    out.write(f"{pid}\trelates_to_{entity_type.lower()}\t{entity_clean}\n")
                for concept in concepts:
                    concept_clean = concept.replace(" ", "_").replace("\n", "_")
                    out.write(f"{pid}\texplores_concept\t{concept_clean}\n")

        print(f"✅ Converted papers to triples in {output_file}")
    except Exception as e:
        print(f"❌ Error converting to triples: {e}")
        raise

# ----------------------------
# STEP 4: Visualize Graph with Clustering and Legend
# ----------------------------
def visualize_clustered_graph(triples_file="graph_triples.txt", title="Knowledge Graph"):
    try:
        G = nx.DiGraph()

        # Read triples and build the graph
        with open(triples_file, encoding="utf-8") as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                G.add_node(h, node_type='paper' if any(x in h for x in ['_', ':']) else r.split('_')[-1])
                G.add_node(t, node_type=r.split('_')[-1] if r.startswith("relates_to_") or r.startswith("explores_") else r.split('_')[-1])
                G.add_edge(h, t, label=r)

        # Community detection for clustering
        partition = community_louvain.best_partition(G.to_undirected())
        pos = nx.spring_layout(G, k=0.6, iterations=100)

        # Assign colors based on clusters
        node_colors = [partition[node] for node in G.nodes()]
        plt.figure(figsize=(20, 20))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.RdYlBu, node_size=800, alpha=0.9)
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=1.0, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
        edge_labels = {(h, t): d['label'] for h, t, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

        # Add legend for communities
        unique_communities = set(partition.values())
        legend_patches = [mpatches.Patch(color=plt.cm.RdYlBu(i / max(unique_communities, default=1)), label=f"Community {i+1}") for i in range(len(unique_communities))]
        plt.legend(handles=legend_patches, title="Communities", loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        plt.title(title, fontsize=18, pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("knowledge_graph.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("✅ Graph visualization displayed and saved as knowledge_graph.png")
    except Exception as e:
        print(f"❌ Error visualizing graph: {e}")
        raise

# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":
    try:
        fetch_ai_safety_papers(max_results=5)
        convert_to_triples()
        visualize_clustered_graph("graph_triples.txt")
    except Exception as e:
        print(f"❌ Error in main pipeline: {e}")
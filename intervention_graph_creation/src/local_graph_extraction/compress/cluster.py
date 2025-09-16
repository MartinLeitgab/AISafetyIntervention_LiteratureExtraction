from falkordb import FalkorDB
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.express as px

from config import load_settings

SETTINGS = load_settings()

class Cluster:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)


    def get_clusters(self, n_clusters: int = 3, query: str = "MATCH (n:NODE) RETURN n LIMIT 100"):
        graph = self.db.select_graph(SETTINGS.falkordb.graph)
        res = graph.query(query)
        df_nodes, df_embeds = self.extract_embeds(res)

        self.data = df_nodes.merge(df_embeds, on="id")
        self.data = self.kmeans_clustering(n_clusters=n_clusters)
        self.cluster_report(self.data)
        self.scatter_3d(
            data=self.data[[col for col in self.data.columns if col.startswith('vec_')]],
            clusters=self.data['cluster'],
            title=f'KMeans Clustering (k={n_clusters})'
        )
        return self.data


    def cluster_report(self, df_clusters: pd.DataFrame):
        cluster_table = df_clusters.groupby('cluster').agg({
            'name': lambda x: list(set(x)),
            'type': lambda x: list(set(x)), 
            'concept_category': lambda x: list(set(x)),
            'paper_id': lambda x: list(set(x)),
            'id': lambda x: list(set(x)), 
            'cluster': 'count'  # Count rows per cluster
            }).rename(columns={'cluster': 'count'})

        cluster_table.to_csv('cluster_report.csv')


    def extract_embeds(self, res, embed_key="embedding"):
        rows = []
        ids = []
        embeddings = []

        for record in res.result_set:
            node = record[0]  # assuming first column is the node
            props = dict(node.properties)  # copy properties
            node_id = node.id
            ids.append(node_id)
            embeddings.append(props.pop(embed_key))  # remove embedding from properties
            props["id"] = node_id
            rows.append(props)

        # node info only
        df_nodes = pd.DataFrame(rows)

        # node id + expanded embeddings
        df_embeds = pd.DataFrame(
            embeddings,
            columns=[f"vec_{i}" for i in range(len(embeddings[0]))]
        )
        df_embeds.insert(0, "id", ids)
        return df_nodes, df_embeds


    def kmeans_clustering(self, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
        embed_data = self.data[[col for col in self.data.columns if col.startswith('vec_')]]
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.data['cluster'] = kmeans.fit_predict(embed_data)
        return self.data
    

    def scatter_3d(self, data, clusters, title=None):
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(data)

        df = pd.DataFrame(reduced_data, columns=['Comp1', 'Comp2', 'Comp3'])
        df['Cluster'] = clusters
        
        # Convert Cluster column to categorical type
        df['Cluster'] = df['Cluster'].astype('category')

        # plotly express
        fig = px.scatter_3d(
            df,
            x='Comp1',
            y='Comp2',
            z='Comp3',
            category_orders={'Cluster': sorted(df['Cluster'].unique())},
            color='Cluster',
            labels={'Comp1': 'Comp 1', 'Comp2': 'Comp 2', 'Comp3': 'Comp 3'},
            opacity=0.5,
            range_x = [min(df['Comp1']) * 1.1, max(df['Comp1']) * 1.1],
            range_y = [min(df['Comp2']) * 1.1, max(df['Comp2']) * 1.1],
            range_z = [min(df['Comp3']) * 1.1, max(df['Comp3']) * 1.1],
            title=title
        )

        fig.update_layout(height=600, width=600)
        # fig.show()
        # save fig as html
        fig.write_html("3d_scatter.html")


    def sil_scores(self, X, n_cluster_start, n_cluster_end):
        scores = []
        for i in range(n_cluster_start, n_cluster_end+1):
            kmeans = KMeans(n_clusters = i, random_state=42).fit(X)
            cluster_labels = kmeans.predict(X)
            sil_score = silhouette_score(X, cluster_labels)
            scores.append((i, sil_score))
        return scores

def main():
    cluster = Cluster(data=None)
    df_clusters = cluster.get_clusters(n_clusters=5, query="MATCH (n:NODE) RETURN n LIMIT 100")
    print(df_clusters.head())


if __name__ == "__main__":
    main()
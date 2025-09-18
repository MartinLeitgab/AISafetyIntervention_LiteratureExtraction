# pyright: standard
# TODO move to the appropriate location
from falkordb_bulk_loader.bulk_insert import bulk_insert
from click.testing import CliRunner
from falkordb import FalkorDB
from os.path import join


def upload_test_graph_to_falkordb(
    db: FalkorDB, server_url: str, graph_name: str, test_graph_dir: str = "./test_graph"
):
    """
    Uploads the test graph in ./test_graph to the FalkorDB instance at server_url.
    Throws a RuntimeError if the upload fails.
    """
    runner = CliRunner()
    result = runner.invoke(
        bulk_insert,
        [
            graph_name,
            "--server-url",
            server_url,
            "--nodes-with-label",
            "CONCEPT",
            join(test_graph_dir, "nodes_Concept.csv"),
            "--nodes-with-label",
            "INTERVENTION",
            join(test_graph_dir, "nodes_Intervention.csv"),
            "--nodes-with-label",
            "RATIONALE",
            join(test_graph_dir, "nodes_Rationale.csv"),
            "--nodes-with-label",
            "SOURCE",
            join(test_graph_dir, "nodes_Source.csv"),
            "--relations-with-type",
            "EDGE",
            join(test_graph_dir, "edges_EDGE.csv"),
            "--relations-with-type",
            "FROM",
            join(test_graph_dir, "edges_FROM.csv"),
            "--relations-with-type",
            "HAS_RATIONALE",
            join(test_graph_dir, "edges_HAS_RATIONALE.csv"),
        ],
    )
    if result.exit_code != 0:
        raise RuntimeError(f"bulk_insert failed: {result.output}\n{result.exception}")
    g = db.select_graph(graph_name)
    r = g.query(
        """
    MATCH (n)
    SET n:NODE
    """
    )


if __name__ == "__main__":
    from config import load_settings

    SETTINGS = load_settings()
    upload_test_graph_to_falkordb(
        db=FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port),
        server_url=f"redis://{SETTINGS.falkordb.host}:{SETTINGS.falkordb.port}",
        graph_name=SETTINGS.falkordb.graph,
    )

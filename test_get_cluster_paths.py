# pyright: standard
#TODO put this file in the appropriate location
from falkordb import FalkorDB, Graph, Path
import pytest
from semantic_compression_part_2 import get_cluster_paths, SETTINGS
import docker


def upload_csv_to_docker_image():
    client = docker.from_env()
    containers = client.containers.list(filters={"ancestor": "falkordb/falkordb"})
    if len(containers) == 0:
        raise RuntimeError("No running FalkorDB container found.")

    container = containers[0]  # Assume the first matching container is the one we want
    container.p
@pytest.fixture
def g():
    db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)
    graph = db.select_graph("TEST_GRAPH")

    graph.query("""
    CREATE (n1:NODE {name: 'Node 1', is_tombstone: false})
    CREATE (n2:NODE {name: 'Node 2', is_tombstone: false})
    CREATE (n3:NODE {name: 'Node 3', is_tombstone: false})
    CREATE (n4:NODE {name: 'Node 4', is_tombstone: false})
    CREATE (n5:NODE {name: 'Node 5', is_tombstone: false})
    CREATE (n6:NODE {name: 'Node 6', is_tombstone: false})
    CREATE (n7:NODE {name: 'Node 7', is_tombstone: false})
    CREATE (n8:NODE {name: 'Node 8', is_tombstone: false})
    CREATE (n9:NODE {name: 'Node 9', is_tombstone: false})
    CREATE (n10:NODE {name: 'Node 10', is_tombstone: false})
                

     CREATE (d1:NODE {name: 'Dead Node 1', is_tombstone: false})
            
    MERGE (n1)-[:EDGE {is_tombstone: false}]->(n2)
    MERGE (n2)-[:EDGE {is_tombstone: false}]->(n3)
    MERGE (n3)-[:EDGE {is_tombstone: false}]->(n4)
    MERGE (n1)-[:EDGE {is_tombstone: false}]->(n1)
            

    MERGE (n1)-[:EDGE {is_tombstone: false}]->(n1)
    """)

    try:
        yield graph
    finally:
        try:
            graph.delete()
        except:
            pass


def test_1(g: Graph):
    """Test that all nodes in the cluster are included """

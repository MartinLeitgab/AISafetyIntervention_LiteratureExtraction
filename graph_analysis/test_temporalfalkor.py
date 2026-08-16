from phase2_step1_loadandparse import *  # noqa: F403

current_id = 40000
end_id = 50000
client = connect_falkordb()  # noqa: F405
node_attrs = load_node_attributes(client)  # noqa: F405
print(f"Loaded {len(node_attrs)} nodes")
print(f"URLs: {sum(1 for a in node_attrs.values() if a.get('url'))}")
print(f"Temporal: {sum(1 for a in node_attrs.values() if a.get('first_published'))}")

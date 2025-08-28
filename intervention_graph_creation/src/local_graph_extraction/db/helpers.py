def lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    if isinstance(v, dict):
        items = []
        for k, val in v.items():
            items.append(f"{k}: {lit(val)}")
        return "{" + ", ".join(items) + "}"
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


def label_for(node_type: str) -> str:
    return "Concept" if node_type == "concept" else "Intervention"

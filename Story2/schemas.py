SCHEMA_FULL = {
  "type": "object",
  "required": ["nodes", "edges"],
  "properties": {
    "nodes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "node_name",
          "node_description",
          "isIntervention"
        ],
        "properties": {
          "node_name": { "type": "string" },
          "node_description": { "type": "string" },
          "isIntervention": { "type": "boolean" },
          "pipeline_stage": { "type": "integer" },
          "maturity_level": { "type": "integer" },
          "implemented": { "type": "boolean" },
        }
      }
    },
    "edges": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "edge_name",
          "edge_description",
          "source_node",
          "target_node",
          "confidence"
        ],
        "properties": {
          "edge_name": { "type": "string" },
          "edge_description": { "type": "string" },
          "source_node": {
            "type": "array",
            "items": { "type": "string" }
          },
          "target_node": {
            "type": "array",
            "items": { "type": "string" }
          },
          "confidence": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
          }
        }
      }
    }
  }
}

SCHEMA_CONCEPTS = {
  "type": "object",
  "required": ["nodes"],
  "properties": {
    "nodes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "node_name",
          "node_description",
          "isIntervention"
        ],
        "properties": {
          "node_name": { "type": "string" },
          "node_description": { "type": "string" },
          "isIntervention": { "type": "boolean" },
        }
      }
    }
  }
}

SCHEMA_INTERVENTIONS = {
  "type": "object",
  "required": ["nodes"],
  "properties": {
    "nodes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "node_name",
          "node_description",
          "isIntervention"
        ],
        "properties": {
          "node_name": { "type": "string" },
          "node_description": { "type": "string" },
          "isIntervention": {
            "type": "integer",
            "enum": [0, 1]
          },
          "pipeline_stage": { "type": "integer" },
          "maturity_level": { "type": "integer" },
          "implemented": { "type": "boolean" },
        }
      }
    }
  }
}

SCHEMA_EDGES = {
  "type": "object",
  "required": ["edges"],
  "properties": {
    "edges": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "edge_name",
          "edge_description",
          "source_node",
          "target_node",
          "confidence"
        ],
        "properties": {
          "edge_name": { "type": "string" },
          "edge_description": { "type": "string" },
          "source_node": {
            "type": "array",
            "items": { "type": "string" }
          },
          "target_node": {
            "type": "array",
            "items": { "type": "string" }
          },
          "confidence": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
          }
        }
      }
    }
  }
}
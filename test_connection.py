# test_connection.py
print("Testing FalkorDB connection...")

try:
    print("1. Loading config...")
    from config import load_settings

    settings = load_settings()
    print("   ✓ Config loaded")
    print(f"   - Host: {settings.falkordb.host}")
    print(f"   - Port: {settings.falkordb.port}")
    print(f"   - Graph: {settings.falkordb.graph}")
except Exception as e:
    print(f"   ✗ Config failed: {e}")
    exit(1)

try:
    print("2. Importing FalkorDB...")
    from falkordb import FalkorDB

    print("   ✓ FalkorDB imported")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

try:
    print("3. Connecting to FalkorDB...")
    db = FalkorDB(host=settings.falkordb.host, port=settings.falkordb.port)
    print("   ✓ FalkorDB connection created")
except Exception as e:
    print(f"   ✗ Connection failed: {e}")
    exit(1)

try:
    print("4. Testing graph selection...")
    g = db.select_graph(settings.falkordb.graph)
    print("   ✓ Graph selected")
except Exception as e:
    print(f"   ✗ Graph selection failed: {e}")
    exit(1)

try:
    print("5. Testing simple query...")
    result = g.ro_query("RETURN 1 AS test")
    print(f"   ✓ Query successful: {result.result_set}")
except Exception as e:
    print(f"   ✗ Query failed: {e}")
    exit(1)

print("All tests passed! FalkorDB is working.")

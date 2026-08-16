#!/usr/bin/env python3
"""
Quick test to verify matplotlib Agg backend works
"""

# CRITICAL: Set backend first
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

print(f"✓ Matplotlib backend: {matplotlib.get_backend()}")

# Test simple plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Test Plot")

# Save to file
plt.savefig("/tmp/test_matplotlib.png", dpi=100)
plt.close()

print("✓ Successfully created test plot at /tmp/test_matplotlib.png")
print("✓ Matplotlib Agg backend is working correctly")

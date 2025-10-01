#!/usr/bin/env python3
"""Quick test to see what Cellpose model.eval() actually returns"""

import numpy as np
from cellpose import models
from PIL import Image

# Load a small test image
img_path = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10/sample1/5/CLLSaSa_07292025_1to10_40min_1_05_raw.jpg"
img = np.array(Image.open(img_path).convert('L'))

# Crop to tiny region for fast testing
test_img = img[0:100, 0:100]

print("Testing Cellpose return values...")
print(f"Test image shape: {test_img.shape}")

# Initialize model
model = models.CellposeModel(gpu=False, model_type='cyto2')

print("Running eval()...")
result = model.eval(
    x=test_img,
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    channels=[0, 0]
)

print(f"\nResult type: {type(result)}")
print(f"Number of return values: {len(result)}")
print(f"Types of each value: {[type(r) for r in result]}")

if len(result) == 3:
    print("\n✓ Returns 3 values: masks, flows, styles")
elif len(result) == 4:
    print("\n✓ Returns 4 values: masks, flows, styles, diams")
else:
    print(f"\n⚠️  Unexpected: returns {len(result)} values")

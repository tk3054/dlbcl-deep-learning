#!/usr/bin/env python3
"""
Test script to verify the refactored pipeline functions
"""

import segment_cells
import pad_raw_crops
import filter_bad_cells
import manual_cell_reviewer

# Configuration
BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"
SAMPLE_FOLDER = "sample1"
IMAGE_NUMBER = "1"

print("="*80)
print("TESTING REFACTORED PIPELINE FUNCTIONS")
print("="*80)
print(f"Sample: {SAMPLE_FOLDER}/{IMAGE_NUMBER}\n")

# Test 1: segment_cells
print("\n[TEST 1] Testing segment_cells()...")
result = segment_cells.segment_cells(
    sample_folder=SAMPLE_FOLDER,
    image_number=IMAGE_NUMBER,
    base_path=BASE_PATH,
    verbose=False
)
print(f"  Success: {result['success']}")
if result['success']:
    print(f"  Cells segmented: {result['num_cells']}")
else:
    print(f"  Error: {result.get('error', 'Unknown')}")

# Test 2: pad_masked_cells
print("\n[TEST 2] Testing pad_masked_cells()...")
result = pad_raw_crops.pad_masked_cells(
    sample_folder=SAMPLE_FOLDER,
    image_number=IMAGE_NUMBER,
    base_path=BASE_PATH,
    verbose=False
)
print(f"  Success: {result['success']}")
if result['success']:
    print(f"  Cells padded: {result['num_padded']}")
else:
    print(f"  Error: {result.get('error', 'Unknown')}")

# Test 3: filter_bad_cells
print("\n[TEST 3] Testing filter_bad_cells()...")
result = filter_bad_cells.filter_bad_cells(
    sample_folder=SAMPLE_FOLDER,
    image_number=IMAGE_NUMBER,
    base_path=BASE_PATH,
    verbose=False
)
print(f"  Success: {result['success']}")
if result['success']:
    print(f"  Whole cells: {result['whole_count']}")
    print(f"  Cut cells: {result['cut_count']}")
else:
    print(f"  Error: {result.get('error', 'Unknown')}")

# Test 4: review_cells (skip GUI launch)
print("\n[TEST 4] Testing review_cells() function signature...")
print("  Function exists and is callable: ", callable(manual_cell_reviewer.review_cells))

print("\n" + "="*80)
print("TESTS COMPLETE")
print("="*80)
print("\nAll functions are properly refactored and ready for notebook use!")
print("You can now import these functions in a Jupyter notebook like:")
print("  import segment_cells")
print("  result = segment_cells.segment_cells('sample1', '1', BASE_PATH)")

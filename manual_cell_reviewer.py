#!/usr/bin/env python3
"""
Interactive Manual Cell Reviewer
Launch a GUI window to manually classify cells as GOOD or BAD
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
from pathlib import Path
import json
import sys


# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Change these values to review different samples (must match previous scripts)
SAMPLE_FOLDER = "sample1"  # Options: "sample1", "sample2", "sample3"
IMAGE_NUMBER = "4"         # Options: "1", "2", "3", "4", etc.

# Auto-generated paths
BASE_PATH = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10"
BASE_DIR = f"{BASE_PATH}/{SAMPLE_FOLDER}/{IMAGE_NUMBER}"
RAW_CROPS_DIR = f"{BASE_DIR}/raw_crops"

class CellReviewer:
    def __init__(self, results, raw_crops_dir, base_dir):
        self.results = sorted(results, key=lambda r: int(r['filename'].replace('cell_', '').replace('.tif', '')))
        self.raw_crops_dir = Path(raw_crops_dir)
        self.base_dir = base_dir
        self.current_idx = 0
        self.classifications = []
        self.finished = False

        # Create main window
        self.root = tk.Tk()
        self.root.title("Cell Quality Reviewer")
        self.root.geometry("800x850")

        # Progress label
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 14, "bold"))
        self.progress_label.pack(pady=10)

        # Image canvas
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg='black')
        self.canvas.pack(pady=10)

        # Info label
        self.info_label = tk.Label(self.root, text="", font=("Arial", 12), justify='left')
        self.info_label.pack(pady=5)

        # Instructions
        instructions = tk.Label(
            self.root,
            text="← Left Arrow: BAD    → Right Arrow: GOOD    B: Undo    Q/Escape: Quit & Save",
            font=("Arial", 11, "bold"),
            fg="blue"
        )
        instructions.pack(pady=10)

        # Bind keyboard
        self.root.bind('<Left>', lambda e: self.classify('BAD'))
        self.root.bind('<Right>', lambda e: self.classify('GOOD'))
        self.root.bind('b', lambda e: self.undo())
        self.root.bind('B', lambda e: self.undo())
        self.root.bind('q', lambda e: self.quit())
        self.root.bind('Q', lambda e: self.quit())
        self.root.bind('<Escape>', lambda e: self.quit())

        # Show first image
        self.show_current()

    def show_current(self):
        if self.current_idx >= len(self.results):
            self.finish()
            return

        result = self.results[self.current_idx]

        # Update progress
        good_count = sum(1 for c in self.classifications if c['classification'] == 'GOOD')
        bad_count = sum(1 for c in self.classifications if c['classification'] == 'BAD')
        progress_text = f"Cell {self.current_idx + 1}/{len(self.results)}  |  ✓ {good_count} GOOD  |  ✗ {bad_count} BAD"
        self.progress_label.config(text=progress_text)

        # Load image
        cell_num = result['filename'].replace('cell_', '').replace('.tif', '')
        matching_files = list(self.raw_crops_dir.glob(f"*cell_{cell_num}_*.tif"))
        if not matching_files:
            matching_files = list(self.raw_crops_dir.glob(f"cell_{cell_num}.tif"))

        if matching_files:
            img = Image.open(matching_files[0])
            # Resize to fit canvas while maintaining aspect ratio
            img.thumbnail((600, 600), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)

            # Center image on canvas
            self.canvas.delete("all")
            x = (600 - img.width) // 2
            y = (600 - img.height) // 2
            self.canvas.create_image(x, y, anchor='nw', image=self.photo)

        # Update info
        edge_pixels = result['edge_consecutive_pixels']
        auto_status = "CUT" if result['is_cut_off'] else "WHOLE"
        info_text = result['filename'] + "\n"
        info_text += f"Edge Pixels: T{edge_pixels['top']} / B{edge_pixels['bottom']} / L{edge_pixels['left']} / R{edge_pixels['right']}\n"
        info_text += f"Auto-detected: {auto_status}"
        self.info_label.config(text=info_text)

    def classify(self, classification):
        if self.current_idx < len(self.results) and not self.finished:
            result = self.results[self.current_idx]
            self.classifications.append({
                'filename': result['filename'],
                'classification': classification,
                'edge_consecutive_pixels': result['edge_consecutive_pixels'],
                'auto_detected': result['is_cut_off']
            })
            self.current_idx += 1
            self.show_current()

    def undo(self):
        if self.classifications and not self.finished:
            # Remove last classification
            self.classifications.pop()
            # Go back one image
            self.current_idx = max(0, self.current_idx - 1)
            self.show_current()

    def finish(self):
        if self.finished:
            return
        self.finished = True

        # Unbind all keys to prevent further input
        self.root.unbind('<Left>')
        self.root.unbind('<Right>')
        self.root.unbind('b')
        self.root.unbind('B')
        self.root.unbind('q')
        self.root.unbind('Q')
        self.root.unbind('<Escape>')

        # Save results
        if self.classifications:
            df_manual = pd.DataFrame(self.classifications)
            output_path = f"{self.base_dir}/manual_classifications.csv"
            df_manual.to_csv(output_path, index=False)
            print(f"\n✅ Manual classifications saved to: {output_path}")

            good_count = sum(1 for c in self.classifications if c['classification'] == 'GOOD')
            bad_count = len(self.classifications) - good_count
            print(f"📊 Summary: {good_count} GOOD, {bad_count} BAD (out of {len(self.classifications)} reviewed)")

        # Show completion message
        self.canvas.delete("all")
        self.info_label.config(text="✅ All done! Closing window...", fg="green")
        self.progress_label.config(text="Review Complete!", fg="green")

        # Close window after short delay
        self.root.after(1000, self.root.quit)

    def quit(self):
        if not self.finished:
            self.finish()

    def run(self):
        try:
            self.root.mainloop()
        finally:
            try:
                self.root.destroy()
            except:
                pass


def load_results_from_csv(base_dir):
    """Load analysis results from CSV file"""
    csv_path = Path(base_dir) / "raw_crops_quality_analysis.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # Convert back to results format
    results = []
    for _, row in df.iterrows():
        results.append({
            'filename': row['filename'],
            'cell_count': row['cell_count'],
            'is_cut_off': row['is_cut_off'],
            'cut_off_edges': eval(row['cut_off_edges']) if pd.notna(row['cut_off_edges']) else [],
            'edge_consecutive_pixels': eval(row['edge_consecutive_pixels']),
            'mask_shape': eval(row['mask_shape'])
        })

    return results


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def review_cells(sample_folder, image_number, base_path, verbose=True):
    """
    Launch GUI for manual cell classification.

    This is the main function to be called from notebooks or other scripts.

    Args:
        sample_folder: Sample folder name (e.g., "sample1", "sample2")
        image_number: Image number within sample (e.g., "1", "2", "3")
        base_path: Base directory path (e.g., "/path/to/data")
        verbose: Print progress messages

    Returns:
        dict with keys:
            - 'classifications': List of manual classifications
            - 'good_count': Number of cells marked as GOOD
            - 'bad_count': Number of cells marked as BAD
            - 'output_csv': Path to output CSV file
            - 'success': Boolean indicating success
            - 'error': Error message if success is False
    """
    # Build paths
    base_dir = f"{base_path}/{sample_folder}/{image_number}"
    raw_crops_dir = f"{base_dir}/raw_crops"

    if verbose:
        print("\n" + "="*60)
        print("MANUAL CELL REVIEW")
        print("="*60)
        print(f"Sample: {sample_folder}/{image_number}")
        print(f"Directory: {base_dir}")
        print("="*60)

    # Load results
    if verbose:
        print("\nLoading analysis results...")

    results = load_results_from_csv(base_dir)
    if results is None:
        return {
            'success': False,
            'error': f"Could not find raw_crops_quality_analysis.csv in {base_dir}",
            'classifications': [],
            'good_count': 0,
            'bad_count': 0
        }

    if verbose:
        print(f"  Loaded {len(results)} cells\n")
        print("A window will pop up. Use arrow keys to classify:")
        print("  <- Left Arrow  = Mark as BAD")
        print("  -> Right Arrow = Mark as GOOD")
        print("  B             = Undo last classification")
        print("  Q or Escape   = Quit and save")
        print("="*60 + "\n")

    # Launch reviewer
    reviewer = CellReviewer(results, raw_crops_dir, base_dir)
    reviewer.run()

    # Get results
    classifications = reviewer.classifications
    good_count = sum(1 for c in classifications if c['classification'] == 'GOOD')
    bad_count = len(classifications) - good_count
    output_csv = f"{base_dir}/manual_classifications.csv"

    if verbose:
        print(f"\nManual review complete:")
        print(f"  GOOD cells: {good_count}")
        print(f"  BAD cells: {bad_count}")
        print(f"  CSV saved: {output_csv}")

    return {
        'success': True,
        'classifications': classifications,
        'good_count': good_count,
        'bad_count': bad_count,
        'output_csv': output_csv,
        'base_dir': base_dir
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MANUAL CELL REVIEW")
    print("="*60)
    print(f"Sample: {SAMPLE_FOLDER}/{IMAGE_NUMBER}")
    print(f"Directory: {BASE_DIR}")
    print("="*60)

    # Load results
    print("\nLoading analysis results...")
    results = load_results_from_csv(BASE_DIR)

    if results is None:
        print(f"ERROR: Could not find raw_crops_quality_analysis.csv in {BASE_DIR}")
        print("Please run filter_bad_cells.py first!")
        sys.exit(1)

    print(f"  Loaded {len(results)} cells\n")

    print("A window will pop up. Use arrow keys to classify:")
    print("  <- Left Arrow  = Mark as BAD")
    print("  -> Right Arrow = Mark as GOOD")
    print("  B             = Undo last classification")
    print("  Q or Escape   = Quit and save")
    print("="*60 + "\n")

    # Launch reviewer
    reviewer = CellReviewer(results, RAW_CROPS_DIR, BASE_DIR)
    reviewer.run()
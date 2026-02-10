#!/usr/bin/env python3
"""
Interactive Cell Sorter GUI
Displays full microscope image with cell boundaries and allows manual classification
"""

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from skimage.segmentation import find_boundaries
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL"


# ============================================================================
# CELL SORTER GUI
# ============================================================================

class CellSorterGUI:
    def __init__(self, base_path):
        self.base_path = Path(base_path)

        self.current_idx = 0
        self.classifications = {}  # unique_id -> classification

        # Discover all samples and images
        print("Discovering samples and images...")
        self.discover_samples_and_images()

        # Load data
        print("Loading data...")
        self.load_all_images_and_masks()
        self.load_cell_data()

        # Create GUI
        self.create_gui()

    def discover_samples_and_images(self):
        """Find all sample folders and their image folders"""
        # Get all sample directories
        all_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]

        # Filter for sample directories (ones that start with 'sample')
        sample_dirs = sorted([d for d in all_dirs if d.name.startswith('sample')])

        if not sample_dirs:
            raise ValueError(f"No sample folders found in {self.base_path}")

        # Discover images in each sample
        self.sample_image_map = {}  # sample_name -> [image_folders]

        for sample_dir in sample_dirs:
            sample_name = sample_dir.name

            # Find image folders in this sample
            image_dirs = [d for d in sample_dir.iterdir() if d.is_dir()]
            image_folders = []

            for img_dir in sorted(image_dirs):
                if (img_dir / "cell_rois").exists():
                    image_folders.append(img_dir.name)

            if image_folders:
                self.sample_image_map[sample_name] = image_folders
                print(f"  {sample_name}: {len(image_folders)} images")

        if not self.sample_image_map:
            raise ValueError(f"No image folders with cell_rois found in any sample")

        total_images = sum(len(imgs) for imgs in self.sample_image_map.values())
        print(f"  Total: {len(self.sample_image_map)} samples, {total_images} images")

    def load_all_images_and_masks(self):
        """Load all images and ROI masks from all samples"""
        self.images_data = []  # List of dicts: {sample, image_number, raw_img, labeled_mask, boundaries, cell_bboxes}
        self.all_cells = []  # Flat list of all cells with sample and image info

        for sample_name, image_folders in self.sample_image_map.items():
            print(f"  Loading {sample_name}...")

            for image_number in image_folders:
                base_dir = self.base_path / sample_name / image_number

                # Load original image
                actin_tif = base_dir / "Actin-FITC.tif"
                raw_files = list(base_dir.glob("*_raw.jpg"))

                if actin_tif.exists():
                    # Load TIF with full bit depth
                    raw_img = cv2.imread(str(actin_tif), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
                    # Normalize to 8-bit for display
                    raw_img = raw_img.astype(float)
                    img_min, img_max = raw_img.min(), raw_img.max()
                    if img_max > img_min:
                        raw_img = (raw_img - img_min) / (img_max - img_min) * 255
                    raw_img = raw_img.astype(np.uint8)
                elif raw_files:
                    raw_img = np.array(Image.open(raw_files[0]).convert('L'))
                else:
                    print(f"    ⚠️  No image found for {sample_name}/{image_number}, skipping")
                    continue

                # Load all ROI masks and create labeled mask
                roi_dir = base_dir / "cell_rois"
                roi_files = sorted(roi_dir.glob("*.tif"))

                labeled_mask = np.zeros(raw_img.shape, dtype=np.uint16)
                cell_bboxes = []

                for roi_file in roi_files:
                    cell_id = int(roi_file.stem.replace('cell_', '').replace('_0', ''))
                    roi_mask = np.array(Image.open(roi_file).convert('L'))

                    # Label this cell
                    labeled_mask[roi_mask > 0] = cell_id

                    # Calculate bounding box
                    coords = np.where(roi_mask > 0)
                    if len(coords[0]) > 0:
                        y_min, y_max = coords[0].min(), coords[0].max()
                        x_min, x_max = coords[1].min(), coords[1].max()
                        bbox = {
                            'sample': sample_name,
                            'image_number': image_number,
                            'cell_id': cell_id,
                            'x_min': x_min,
                            'y_min': y_min,
                            'x_max': x_max,
                            'y_max': y_max
                        }
                        cell_bboxes.append(bbox)
                        self.all_cells.append(bbox)

                # Create boundaries
                boundaries = find_boundaries(labeled_mask, mode='inner')

                # Store image data
                self.images_data.append({
                    'sample': sample_name,
                    'image_number': image_number,
                    'raw_img': raw_img,
                    'labeled_mask': labeled_mask,
                    'boundaries': boundaries,
                    'cell_bboxes': cell_bboxes
                })

            print(f"    ✓ {sum(len(d['cell_bboxes']) for d in self.images_data if d['sample'] == sample_name)} cells")

        print(f"  Total: {len(self.all_cells)} cells across {len(self.images_data)} images from {len(self.sample_image_map)} samples")

    def load_cell_data(self):
        """Load cell measurements from CSV"""
        # Try to load all_samples_combined.csv first
        all_samples_csv = self.base_path / "all_samples_combined.csv"

        if all_samples_csv.exists():
            self.cell_data = pd.read_csv(all_samples_csv)
            print(f"  Loaded measurements for {len(self.cell_data)} cells from all_samples_combined.csv")
        else:
            # Fall back to loading individual sample CSVs
            print("  ⚠️  No all_samples_combined.csv found, loading individual sample CSVs...")
            dfs = []
            for sample_name in self.sample_image_map.keys():
                csv_file = self.base_path / sample_name / "combined_measurements.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    dfs.append(df)

            if dfs:
                self.cell_data = pd.concat(dfs, ignore_index=True)
                print(f"  Loaded measurements for {len(self.cell_data)} cells from {len(dfs)} sample CSVs")
            else:
                print("  ⚠️  No measurement data found")
                self.cell_data = None

    def create_gui(self):
        """Create the GUI window"""
        self.root = tk.Tk()
        self.root.title(f"Cell Sorter - All Samples")

        # Calculate window size based on first image (assume all same size)
        if self.images_data:
            img_height, img_width = self.images_data[0]['raw_img'].shape
            max_width = 1200
            max_height = 900

            # Scale if needed
            scale = min(max_width / img_width, max_height / img_height, 1.0)
            self.display_width = int(img_width * scale)
            self.display_height = int(img_height * scale)
        else:
            self.display_width = 1200
            self.display_height = 900

        # Progress label
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 14, "bold"))
        self.progress_label.pack(pady=10)

        # Image canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.display_width,
            height=self.display_height,
            bg='black'
        )
        self.canvas.pack(pady=10)

        # Info label
        self.info_label = tk.Label(self.root, text="", font=("Arial", 11), justify='left')
        self.info_label.pack(pady=5)

        # Instructions
        instructions = tk.Label(
            self.root,
            text="← Left: BAD    → Right: GOOD    1-6: Classify    Space: Skip    B: Undo    S: Save    Q/Esc: Quit",
            font=("Arial", 11, "bold"),
            fg="blue"
        )
        instructions.pack(pady=10)

        # Classification options
        self.class_options = {
            '1': 'GOOD',
            '2': 'BAD',
            '3': 'UNCERTAIN',
            '4': 'DEBRIS',
            '5': 'DOUBLET',
            '6': 'EDGE',
        }

        # Bind keyboard
        for key in self.class_options.keys():
            self.root.bind(key, lambda e, k=key: self.classify(self.class_options[k]))

        # Arrow keys for quick classification
        self.root.bind('<Left>', lambda e: self.classify('BAD'))
        self.root.bind('<Right>', lambda e: self.classify('GOOD'))

        self.root.bind('<space>', lambda e: self.skip())
        self.root.bind('b', lambda e: self.undo())
        self.root.bind('B', lambda e: self.undo())
        self.root.bind('s', lambda e: self.save())
        self.root.bind('S', lambda e: self.save())
        self.root.bind('q', lambda e: self.quit())
        self.root.bind('Q', lambda e: self.quit())
        self.root.bind('<Escape>', lambda e: self.quit())

        # Show first cell
        self.show_current()

    def create_image_with_highlight(self):
        """Create image with boundaries and current cell highlighted"""
        if self.current_idx >= len(self.all_cells):
            return None

        # Get current cell and find its image data
        current_cell = self.all_cells[self.current_idx]
        sample_name = current_cell['sample']
        image_number = current_cell['image_number']

        # Find the image data for this cell
        img_data = None
        for data in self.images_data:
            if data['sample'] == sample_name and data['image_number'] == image_number:
                img_data = data
                break

        if img_data is None:
            return None

        # Create RGB image from grayscale
        overlay = np.stack([img_data['raw_img']] * 3, axis=-1).copy()

        # Draw red boundaries
        overlay[img_data['boundaries']] = [255, 0, 0]

        # Convert to PIL Image
        pil_img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_img)

        # Highlight current cell with yellow box
        # Draw thick yellow box
        for offset in range(3):  # 3-pixel thick border
            draw.rectangle(
                [
                    current_cell['x_min'] - offset,
                    current_cell['y_min'] - offset,
                    current_cell['x_max'] + offset,
                    current_cell['y_max'] + offset
                ],
                outline='yellow',
                width=1
            )

        # Add cell number label
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()

        label = f"{sample_name}/{image_number} - Cell {current_cell['cell_id']}"
        draw.text(
            (current_cell['x_min'], current_cell['y_min'] - 25),
            label,
            fill='yellow',
            font=font
        )

        # Resize for display
        pil_img = pil_img.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)

        return pil_img

    def show_current(self):
        """Display current cell"""
        if self.current_idx >= len(self.all_cells):
            self.finish()
            return

        # Update progress
        classified = len(self.classifications)
        remaining = len(self.all_cells) - classified
        progress_text = f"Cell {self.current_idx + 1}/{len(self.all_cells)}  |  Classified: {classified}  |  Remaining: {remaining}"
        self.progress_label.config(text=progress_text)

        # Create and display image
        img = self.create_image_with_highlight()
        if img is None:
            print(f"⚠️  Warning: Could not create image for cell {self.current_idx + 1}, finishing...")
            self.finish()
            return

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(
            self.display_width // 2,
            self.display_height // 2,
            image=self.photo
        )

        # Show cell info
        current_cell = self.all_cells[self.current_idx]
        sample_name = current_cell['sample']
        cell_id = current_cell['cell_id']
        image_number = current_cell['image_number']

        info_lines = [
            f"Sample: {sample_name}",
            f"Image: {image_number}",
            f"Cell ID: {cell_id}",
            f"Position: ({current_cell['x_min']}, {current_cell['y_min']})",
            f"Size: {current_cell['x_max'] - current_cell['x_min']}x{current_cell['y_max'] - current_cell['y_min']} px",
        ]

        # Add measurement data if available
        if self.cell_data is not None:
            # Match by sample, image, and cell_id
            numeric_image = image_number.replace('[CAR]', '').replace('[', '').replace(']', '')
            cell_row = self.cell_data[
                (self.cell_data['sample'] == sample_name) &
                (self.cell_data['image'] == int(numeric_image)) &
                (self.cell_data['cell_id'] == cell_id)
            ]
            if not cell_row.empty:
                info_lines.append(f"Area: {cell_row.iloc[0]['area']:.1f}")
                if 'actin_median' in cell_row.columns:
                    info_lines.append(f"Actin intensity: {cell_row.iloc[0]['actin_median']:.1f}")

        # Add classification options
        info_lines.append("\nClassifications:")
        for key, label in self.class_options.items():
            info_lines.append(f"  {key}: {label}")

        self.info_label.config(text="\n".join(info_lines))

    def classify(self, classification):
        """Classify current cell and move to next"""
        current_cell = self.all_cells[self.current_idx]
        sample_name = current_cell['sample']
        image_number = current_cell['image_number']
        cell_id = current_cell['cell_id']

        # Get unique_id
        if self.cell_data is not None:
            # Try to extract numeric image_number for matching
            numeric_image = image_number.replace('[CAR]', '').replace('[', '').replace(']', '')
            cell_row = self.cell_data[
                (self.cell_data['sample'] == sample_name) &
                (self.cell_data['image'] == int(numeric_image)) &
                (self.cell_data['cell_id'] == cell_id)
            ]
            if not cell_row.empty and 'unique_id' in cell_row.columns:
                unique_id = cell_row.iloc[0]['unique_id']
            else:
                unique_id = f"{sample_name}_{image_number}_{cell_id}"
        else:
            unique_id = f"{sample_name}_{image_number}_{cell_id}"

        self.classifications[unique_id] = {
            'classification': classification,
            'timestamp': datetime.now().isoformat()
        }

        self.current_idx += 1
        self.show_current()

    def skip(self):
        """Skip current cell without classifying"""
        self.current_idx += 1
        self.show_current()

    def undo(self):
        """Go back to previous cell"""
        if self.current_idx > 0:
            self.current_idx -= 1
            # Remove classification if it exists
            current_cell = self.all_cells[self.current_idx]

            # Try to find unique_id to remove
            if self.cell_data is not None:
                numeric_image = current_cell['image_number'].replace('[CAR]', '').replace('[', '').replace(']', '')
                cell_row = self.cell_data[
                    (self.cell_data['sample'] == current_cell['sample']) &
                    (self.cell_data['image'] == int(numeric_image)) &
                    (self.cell_data['cell_id'] == current_cell['cell_id'])
                ]
                if not cell_row.empty and 'unique_id' in cell_row.columns:
                    unique_id = cell_row.iloc[0]['unique_id']
                else:
                    unique_id = f"{current_cell['sample']}_{current_cell['image_number']}_{current_cell['cell_id']}"
            else:
                unique_id = f"{current_cell['sample']}_{current_cell['image_number']}_{current_cell['cell_id']}"

            if unique_id in self.classifications:
                del self.classifications[unique_id]
            self.show_current()

    def save(self):
        """Save classifications to CSV"""
        if not self.classifications:
            print("⚠️  No classifications to save yet")
            return

        # Create dataframe
        data = []
        for unique_id, info in self.classifications.items():
            data.append({
                'unique_id': unique_id,
                'classification': info['classification'],
                'timestamp': info['timestamp']
            })

        df = pd.DataFrame(data)

        # Save to base directory (all samples)
        output_file = self.base_path / "cell_classifications.csv"
        df.to_csv(output_file, index=False)

        print(f"💾 Saved {len(data)} classifications to {output_file}")

        # Show brief save confirmation in UI (preserve current info)
        current_text = self.info_label.cget("text")
        self.info_label.config(text=f"💾 SAVED {len(data)} classifications!\n\n{current_text}")
        self.root.after(1500, lambda: self.info_label.config(text=current_text))

    def finish(self):
        """Finish classification session"""
        self.save()

        # Show completion message
        self.canvas.delete('all')
        self.canvas.create_text(
            self.display_width // 2,
            self.display_height // 2,
            text=f"Classification Complete!\n\n{len(self.classifications)} cells classified",
            font=("Arial", 24, "bold"),
            fill='green'
        )

        self.progress_label.config(text="✓ COMPLETE")

    def quit(self):
        """Save and quit"""
        self.save()
        self.root.destroy()

    def run(self):
        """Start the GUI"""
        self.root.mainloop()


# ============================================================================
# MAIN
# ============================================================================

def main():
    sorter = CellSorterGUI(
        base_path=BASE_PATH
    )
    sorter.run()


if __name__ == "__main__":
    main()

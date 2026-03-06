import argparse
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def _collect_tif_files(input_path: Path) -> list[Path]:
    tif_files = []
    tif_files.extend(input_path.glob("*.tif"))
    tif_files.extend(input_path.glob("*.tiff"))
    tif_files.extend(input_path.glob("*.TIF"))
    tif_files.extend(input_path.glob("*.TIFF"))
    return sorted(tif_files)


def _compute_patient_min_max(tif_files: list[Path]) -> tuple[float, float]:
    patient_min = None
    patient_max = None

    for tif_path in tqdm(tif_files, desc="Pass 1/2: scan patient min/max"):
        img = tiff.imread(tif_path)
        img_min = float(np.min(img))
        img_max = float(np.max(img))

        if patient_min is None or img_min < patient_min:
            patient_min = img_min
        if patient_max is None or img_max > patient_max:
            patient_max = img_max

    if patient_min is None or patient_max is None:
        raise RuntimeError("No pixel data found while scanning images.")

    return patient_min, patient_max


def normalize_tif_batch(input_folder, output_folder="normalized_tif", hist_bins=512):
    """
    Normalize all TIF images in a folder using one patient-wide min/max
    computed from all pixels across all images. Also writes a histogram
    visualizing the distribution of all patient pixels.
    
    Parameters:
    -----------
    input_folder : str
        Input folder path
    output_folder : str
        Output folder name
    hist_bins : int
        Number of bins for global pixel distribution histogram.
    """
    input_path = Path(input_folder)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_path}")

    output_path = input_path / output_folder
    output_path.mkdir(exist_ok=True)

    tif_files = _collect_tif_files(input_path)
    if not tif_files:
        raise RuntimeError(f"No TIF files found in folder: {input_path}")

    print(f"Found {len(tif_files)} TIF files")
    patient_min, patient_max = _compute_patient_min_max(tif_files)
    print(f"Patient-wide raw min: {patient_min}")
    print(f"Patient-wide raw max: {patient_max}")

    if patient_max == patient_min:
        raise RuntimeError(
            "Aborting normalization: patient-wide max equals min "
            f"({patient_max}). All pixels are constant, which indicates bad input."
        )

    hist_counts = np.zeros(hist_bins, dtype=np.int64)
    hist_edges = np.linspace(patient_min, patient_max, hist_bins + 1, dtype=np.float64)

    print("Output format: 32-bit float (range from patient-wide raw min/max)")
    for tif_path in tqdm(tif_files, desc="Pass 2/2: normalize + histogram"):
        try:
            img = tiff.imread(tif_path)
            img_float = img.astype(np.float32)

            counts, _ = np.histogram(img_float.ravel(), bins=hist_edges)
            hist_counts += counts

            output_array = (img_float - patient_min) / (patient_max - patient_min)

            output_filename = f"{tif_path.stem}_normalized.tif"
            output_file = output_path / output_filename
            tiff.imwrite(output_file, output_array)

        except Exception as e:
            print(f"\nError processing {tif_path.name}: {e}")

    hist_plot_path = output_path / "patient_pixel_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stairs(hist_counts, hist_edges, linewidth=1.5)
    ax.set_title("Patient-wide Pixel Distribution (All Images, All Pixels)")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Pixel Count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(hist_plot_path, dpi=150)
    plt.close(fig)

    print(f"\nComplete! Normalized images saved to: {output_path}")
    print(f"Distribution plot saved to: {hist_plot_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize all TIFs in a folder using patient-wide raw min/max."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to folder containing TIF images for one patient.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="normalized_tif",
        help="Output subfolder under input folder (default: normalized_tif).",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=512,
        help="Histogram bin count for the all-pixel distribution plot.",
    )
    return parser

if __name__ == "__main__":
    # Set your own paths/params here (direct edit mode).
    INPUT_FOLDER = "/path/to/patient/folder"
    OUTPUT_FOLDER = "normalized_tif"
    HIST_BINS = 512

    # If INPUT_FOLDER is still the placeholder, fall back to CLI args.
    if INPUT_FOLDER == "/path/to/patient/folder":
        args = _build_parser().parse_args()
        normalize_tif_batch(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            hist_bins=args.hist_bins,
        )
    else:
        normalize_tif_batch(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            hist_bins=HIST_BINS,
        )

#!/usr/bin/env python3
"""
Plot Intensity Histograms for All Patients
Runs plotAvgIntensity.py on each patient folder within responder/non-responder categories.

Usage:
    python plotAvgIntensity_allPatients.py
"""

from pathlib import Path
from plotAvgIntensity import plot_all_intensity_histograms

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path("/Users/taeeonkong/Desktop/DL Project")
CATEGORY = "non-responder"  # "responder" or "non-responder"
# =============================================================================


def run_all_patients():
    """Run intensity histogram plotting for all patients in the category."""
    category_path = BASE_DIR / CATEGORY

    if not category_path.exists():
        print(f"Error: {category_path} does not exist")
        return

    print(f"Processing {CATEGORY.upper()} patients...")
    print(f"Base directory: {BASE_DIR}")
    print("=" * 60)

    results = []

    # Iterate through each patient folder
    for patient_folder in sorted(category_path.iterdir()):
        if patient_folder.is_dir() and not patient_folder.name.startswith('.'):
            patient_label = patient_folder.name
            print(f"\n>>> Processing: {patient_label}")

            result = plot_all_intensity_histograms(
                base_path=str(patient_folder),
                csv_file="all_samples_combined.csv",
                patient_label=patient_label,
                verbose=True
            )

            results.append({
                'patient': patient_label,
                'success': result['success'],
                'error': result.get('error'),
                'figure_path': result.get('figure_path')
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"✓ Successful: {len(successful)}")
    for r in successful:
        print(f"    - {r['patient']}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for r in failed:
            print(f"    - {r['patient']}: {r['error']}")

    print("\nDone!")


if __name__ == "__main__":
    run_all_patients()

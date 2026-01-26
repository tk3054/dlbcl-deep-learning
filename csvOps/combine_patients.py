import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path("/Users/taeeonkong/Desktop/DL Project")
CATEGORY = "responder"  # "responder" or "non-responder"
# =============================================================================


def combine_patient_csvs(category: str):
    """
    Combine all_samples_combined_classified.csv files from a category folder
    (responder or non-responder) and add a patient_id column.
    """
    category_path = BASE_DIR / category

    if not category_path.exists():
        print(f"Error: {category_path} does not exist")
        return

    all_dfs = []

    # Iterate through each patient folder
    for patient_folder in category_path.iterdir():
        if patient_folder.is_dir() and not patient_folder.name.startswith('.'):
            # Try classified first, then fall back to unclassified
            csv_path = patient_folder / "all_samples_combined_classified.csv"
            if not csv_path.exists():
                csv_path = patient_folder / "all_samples_combined.csv"

            if csv_path.exists():
                print(f"Reading: {csv_path}")
                df = pd.read_csv(csv_path)
                # Add patient_id column with the folder name (e.g., "01-06-2026 DLBCL 118867")
                df.insert(0, 'patient_id', patient_folder.name)
                all_dfs.append(df)
            else:
                print(f"Warning: No CSV found in {patient_folder}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        output_path = category_path / "all_patients_combined.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
        print(f"Total rows: {len(combined_df)}")
    else:
        print("No CSV files found to combine.")


if __name__ == "__main__":
    print(f"Combining {CATEGORY.upper()} data...")
    print(f"Base directory: {BASE_DIR}")
    print("=" * 50)
    combine_patient_csvs(CATEGORY)
    print("\nDone!")
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NORMALIZER="${SCRIPT_DIR}/normalize_per_patient.py"

if [[ ! -f "${NORMALIZER}" ]]; then
  echo "Missing normalizer script: ${NORMALIZER}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

declare -a CHANNEL_DIRS=(
  "formatted_actin"
  "formatted_cd45ra"
  "formatted_ccr7"
)

shopt -s nullglob

declare -a PATIENT_DIRS=()
for cohort in "${PROJECT_ROOT}/responder" "${PROJECT_ROOT}/non-responder"; do
  [[ -d "${cohort}" ]] || continue
  for patient_dir in "${cohort}"/*DLBCL*; do
    [[ -d "${patient_dir}" ]] || continue
    PATIENT_DIRS+=("${patient_dir}")
  done
done

if [[ ${#PATIENT_DIRS[@]} -eq 0 ]]; then
  echo "No patient directories found under responder/ or non-responder/." >&2
  exit 1
fi

for patient_dir in "${PATIENT_DIRS[@]}"; do
  hist_dir="${patient_dir}/DLBCL_processed/histograms"
  mkdir -p "${hist_dir}"

  echo "Processing patient: ${patient_dir}"
  for channel_dir_name in "${CHANNEL_DIRS[@]}"; do
    input_dir="${patient_dir}/${channel_dir_name}"
    [[ -d "${input_dir}" ]] || {
      echo "  Skipping missing folder: ${input_dir}"
      continue
    }

    hist_file="${channel_dir_name}_patient_pixel_distribution.png"
    echo "  Normalizing ${input_dir}"
    "${PYTHON_BIN}" "${NORMALIZER}" \
      "${input_dir}" \
      --output-folder normalized_tif \
      --hist-output-dir "${hist_dir}" \
      --hist-filename "${hist_file}"
  done
done

echo "Normalization complete."

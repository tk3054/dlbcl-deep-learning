"""
Helpers for normalizing pipeline configuration inputs.
"""

def normalize_image_list(values):
    """Convert image filter input to a list of strings."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        values = [values]

    normalized = []
    for val in values:
        if isinstance(val, str):
            stripped = val.strip()
            if stripped:
                normalized.append(stripped)
        elif isinstance(val, (int, float)):
            normalized.append(str(int(val)))
        else:
            normalized.append(str(val))
    return normalized


def normalize_image_filter_config(config):
    """
    Normalize IMAGES_TO_PROCESS into (per-sample map, default list).

    Accepts:
        - None / empty: no filtering
        - dict: {sample_number: [images]}
        - list/tuple/set/single value: applies to all samples
    """
    per_sample = {}
    default_filter = None

    if not config:
        return per_sample, default_filter

    if isinstance(config, dict):
        for key, values in config.items():
            try:
                sample_num = int(key)
            except (TypeError, ValueError):
                continue
            per_sample[sample_num] = normalize_image_list(values)
        return per_sample, default_filter

    default_filter = normalize_image_list(config)
    return per_sample, default_filter


def extract_sample_number(name):
    """Extract numeric sample ID from a folder name like 'sample3'."""
    import re
    match = re.search(r'sample\D*(\d+)', name, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def filter_image_folders(sample_folder, folders, filters_map, filters_default, announce=False):
    """Filter image folders based on per-sample or default filters."""
    sample_number = extract_sample_number(sample_folder)
    allowed = filters_map.get(sample_number)
    if allowed is None:
        allowed = filters_default
    if allowed is None:
        return folders

    if len(allowed) == 0:
        if announce:
            print(
                f"No image restrictions configured for {sample_folder}. "
                "Processing all image folders."
            )
        return folders

    allowed_strings = {str(item) for item in allowed}
    # Support folders like "14[large cell]" when user specifies "14".
    filtered = []
    for folder in folders:
        if folder in allowed_strings:
            filtered.append(folder)
            continue
        # If allowed value is numeric, allow prefix match on folder name.
        # e.g. allowed "14" matches "14[large cell]"
        for allowed_val in allowed_strings:
            if allowed_val.isdigit() and folder.startswith(allowed_val):
                next_char = folder[len(allowed_val):len(allowed_val)+1]
                if next_char == "" or not next_char.isdigit():
                    filtered.append(folder)
                    break

    if announce:
        if filtered:
            def _image_sort_key(value):
                if value.isdigit():
                    return (0, int(value))
                return (1, value)

            print(
                f"Restricting to configured images for {sample_folder}: "
                f"{', '.join(sorted(filtered, key=_image_sort_key))}"
            )
        else:
            print(
                f"Configured image list for {sample_folder} produced no matches. "
                "Skipping this sample."
            )
    return filtered

// ImageJ Macro: Load ROIs (HEADLESS VERSION)
// Loads cell ROIs onto original image in batch mode

// ============================================================================
// CONFIGURATION - Set by Python (do not edit manually)
// ============================================================================

// These values are replaced by Python when running the pipeline
sampleFolder = "SAMPLE_PLACEHOLDER";
imageNumber = "IMAGE_PLACEHOLDER";
basePath = "BASE_PATH_PLACEHOLDER";

// Auto-generated paths
baseDir = basePath + "/" + sampleFolder + "/" + imageNumber + "/";
roi_folder = baseDir + "ROI_DIR_PLACEHOLDER/";  // ROI folder provided by Python
original_image = baseDir + "original_image.tif";


// ============================================================================
// LOAD ROIs
// ============================================================================

// Enable batch mode for headless operation
setBatchMode(true);

// Load All Cell ROIs onto Original Image

// Check if original_image exists (Cellpose should have created it)
// Only create if missing
if (!File.exists(original_image)) {
    actin_fitc = baseDir + "Actin-FITC.tif";
    if (File.exists(actin_fitc)) {
        open(actin_fitc);
        run("Duplicate...", "title=original_image.tif");
        saveAs("Tiff", original_image);
        close();
    } else {
        print("Error: Neither original_image.tif nor Actin-FITC.tif found!");
    }
}

// Open original image
open(original_image);
original_id = getImageID();

// Get list of ROI files
list = getFileList(roi_folder);
roi_count = 0;

// Count ROI files
for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tif")) {
        roi_count++;
    }
}

// Close original image
selectImage(original_id);
close();

// Disable batch mode
setBatchMode(false);

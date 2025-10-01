// ImageJ Macro: Load ROIs (HEADLESS VERSION)
// Loads cell ROIs onto original image in batch mode

// ============================================================================
// CONFIGURATION - EDIT THESE
// ============================================================================

// Change these values to process different samples
sampleFolder = "sample1";  // Options: "sample1", "sample2", "sample3"
imageNumber = "4";         // Options: "1", "2", "3", "4", etc.

// Auto-generated paths
basePath = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10";
baseDir = basePath + "/" + sampleFolder + "/" + imageNumber + "/";
roi_folder = baseDir + "cell_rois/";
original_image = baseDir + "original_image.tif";


// ============================================================================
// LOAD ROIs
// ============================================================================

// Enable batch mode for headless operation
setBatchMode(true);

// Load All Cell ROIs onto Original Image

// Check if original_image exists, if not create it from Actin-FITC
if (!File.exists(original_image)) {
    print("original_image not found, creating from Actin-FITC.tif");
    actin_fitc = baseDir + "Actin-FITC.tif";
    if (File.exists(actin_fitc)) {
        open(actin_fitc);
        run("Duplicate...", "title=original_image.tif");
        saveAs("Tiff", original_image);
        close();
        print("Created " + original_image);
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

print("Loaded " + roi_count + " ROIs");

// Close original image
selectImage(original_id);
close();

// Disable batch mode
setBatchMode(false);
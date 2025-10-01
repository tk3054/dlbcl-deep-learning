// ImageJ Macro: Process Actin-FITC for Cell Analysis (HEADLESS VERSION)
// Automates workflow using your manually thresholded mask
//
// PREREQUISITES:
// 1. Manually threshold Actin-FITC.tif in ImageJ:
//    - Open Actin-FITC.tif
//    - Duplicate (Command+Shift+D)
//    - Smooth (Command+Shift+S)
//    - Threshold (Command+Shift+T) - adjust to your liking
// 2. Save the thresholded duplicate as "duplicate-Actin-FITC-1.tif" in the sample folder
//
// FUNCTIONALITY:
// - Opens your saved duplicate-Actin-FITC-1.tif
// - Saves raw image (from original Actin-FITC.tif)
// - Saves binary mask (from your thresholded duplicate)
// - Can be run from notebook automatically!
//
// Author: Automated from manual protocol

// ============================================================================
// CONFIGURATION - EDIT THESE
// ============================================================================

// Change these values to process different samples
sampleFolder = "sample1";  // Options: "sample1", "sample2", "sample3"
imageNumber = "4";         // Options: "1", "2", "3", "4", etc.
createMask = true;         // Create mask (true for watershed, false for Cellpose)

// Auto-generated path
basePath = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10";
dir = basePath + "/" + sampleFolder + "/" + imageNumber + "/";

// File naming configuration
donorName = "CLLSaSa";
date = "07292025";
stiffness = "1to10";
spreadingTime = "40min";
sampleNumber = "1";
imageNumberFormatted = "0" + imageNumber;  // Add leading zero if needed

// Construct filename prefix
filePrefix = donorName + "_" + date + "_" + stiffness + "_" + spreadingTime + "_" + sampleNumber + "_" + imageNumberFormatted;


// ============================================================================
// MAIN PROCESSING
// ============================================================================

// Enable batch mode for headless operation
setBatchMode(true);

// Open original Actin-FITC
actinPath = dir + "Actin-FITC.tif";
if (!File.exists(actinPath)) {
    print("ERROR: Actin-FITC.tif not found at: " + actinPath);
    exit("Actin-FITC.tif file not found!");
}
open(actinPath);
print("Opened original: " + actinPath);

// Create duplicate for thresholding
selectWindow("Actin-FITC.tif");
run("Duplicate...", "title=thresholded_temp");

// Auto-threshold the duplicate
selectWindow("thresholded_temp");
run("Smooth");
setAutoThreshold("Default dark");
run("Convert to Mask");
print("Auto-thresholded duplicate created");

print("File prefix: " + filePrefix);
print("Image directory: " + dir);

// Step 1: Save raw image (from original Actin-FITC.tif)
print("Step 1: Saving raw image...");
selectWindow("Actin-FITC.tif"); // Use original for raw
run("Duplicate...", "title=" + filePrefix + "_raw_temp");
fullRawPath = dir + filePrefix + "_raw.jpg";
saveAs("Jpeg", fullRawPath);
print("Saved raw to: " + fullRawPath);
close(); // Close raw duplicate

// Step 2: Create and save binary mask (only if createMask is true)
if (createMask) {
    print("Step 2: Creating mask...");
    selectWindow("thresholded_temp");
    run("Duplicate...", "title=" + filePrefix + "_mask_temp");
    fullMaskPath = dir + filePrefix + "_mask.jpg";
    saveAs("Jpeg", fullMaskPath);
    print("Saved mask to: " + fullMaskPath);
    close();
} else {
    print("Step 2: Skipping mask creation (not needed for Cellpose)");
}








print("");
print("✅ Macro completed successfully!");

// Disable batch mode
setBatchMode(false);


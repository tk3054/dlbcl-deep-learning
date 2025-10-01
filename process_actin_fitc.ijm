// ImageJ Macro: Process Actin-FITC for Cell Analysis
// Automates complete workflow from raw image to binary mask
//
// PREREQUISITES:
// 1. Actin-FITC.tif file must exist in the sample/image directory
// 2. Configure sampleFolder and imageNumber variables below
//
// FUNCTIONALITY:
// - Opens Actin-FITC.tif automatically
// - Creates duplicate and applies smoothing
// - Auto-thresholds using default method
// - Saves raw image and binary mask
//
// Author: Automated from manual protocol

// ============================================================================
// CONFIGURATION - EDIT THESE
// ============================================================================

// Change these values to process different samples
sampleFolder = "sample1";  // Options: "sample1", "sample2", "sample3"
imageNumber = "4";         // Options: "1", "2", "3", "4", etc.

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

// Open the Actin-FITC image if not already open
actinPath = dir + "Actin-FITC.tif";
if (!File.exists(actinPath)) {
    print("ERROR: Actin-FITC.tif not found at: " + actinPath);
    exit("Actin-FITC.tif file not found!");
}

// Check if image is already open, otherwise open it
if (nImages == 0 || !isOpen("Actin-FITC.tif")) {
    open(actinPath);
    print("Opened: " + actinPath);
}

// Create the thresholded duplicate
selectWindow("Actin-FITC.tif");
run("Duplicate...", "title=duplicate-Actin-FITC-1.tif");
print("Created duplicate: duplicate-Actin-FITC-1.tif");

// Apply smoothing and thresholding to the duplicate
selectWindow("duplicate-Actin-FITC-1.tif");
run("Smooth");
print("Applied smoothing");

// Auto-threshold using default method
setAutoThreshold("Default dark");
run("Convert to Mask");
print("Applied auto-threshold and converted to mask");

// Get the current image title
originalTitle = getTitle();

// Extract base name - handle duplicate naming
if (startsWith(originalTitle, "duplicate-")) {
    // Remove 'duplicate-' prefix and '-1.tif' suffix
    baseName = substring(originalTitle, 10); // Remove 'duplicate-'
    baseName = replace(baseName, "-1.tif", "");
} else {
    // Fallback: remove .tif extension
    baseName = replace(originalTitle, ".tif", "");
}

print("Processing: " + originalTitle);
print("Base name extracted: " + baseName);
print("File prefix: " + filePrefix);
print("Image directory: " + dir);
print("Starting from pre-thresholded duplicate image...");

// Step 1: Save raw image (from original Actin-FITC.tif)
print("Step 1: Saving raw image...");
selectWindow("Actin-FITC.tif"); // Use original for raw
run("Duplicate...", "title=" + filePrefix + "_raw_temp");
fullRawPath = dir + filePrefix + "_raw.jpg";
saveAs("Jpeg", fullRawPath);
print("Saved raw to: " + fullRawPath);
close(); // Close raw duplicate

// Step 2: Create and save binary mask
print("Step 2: Creating mask...");
selectWindow("duplicate-Actin-FITC-1.tif"); // Use thresholded duplicate for mask
run("Duplicate...", "title=" + filePrefix + "_mask_temp");
run("Flatten"); // Flatten the image first
run("Convert to Mask"); // Convert to actual binary mask
fullPath = dir + filePrefix + "_mask.jpg";
saveAs("Jpeg", fullPath);
print("Saved mask to: " + fullPath);
close(); // Close mask temp

// Step 4: Return to thresholded image for analysis
print("Step 4: Setting up for particle analysis...");
selectWindow("duplicate-Actin-FITC-1.tif"); // Explicitly select the thresholded duplicate








print("");
print("✅ Macro completed successfully!");


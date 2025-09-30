// ImageJ Macro: Process Pre-Thresholded Actin-FITC for Cell Analysis
// Automates workflow starting from manually thresholded image to labeled mask and measurements
//
// PREREQUISITES:
// 1. Open Actin-FITC.tif in ImageJ
// 2. Create duplicate (Command+Shift+D) and name it 'duplicate-Actin-FITC-1.tif'
// 3. Manually smooth the duplicate (Command+Shift+S)
// 4. Manually threshold the duplicate (Command+Shift+T) - adjust as needed
// 5. Then run this macro on the thresholded duplicate
//
// Author: Automated from manual protocol

// ============================================================================
// CONFIGURATION - EDIT THESE
// ============================================================================

// Change these values to process different samples
sampleFolder = "sample1";  // Options: "sample1", "sample2", "sample3"
imageNumber = "3";         // Options: "1", "2", "3", "4", etc.

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


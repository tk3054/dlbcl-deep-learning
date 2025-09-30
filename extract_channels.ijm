// ============================================================================
// CONFIGURATION - EDIT THESE
// ============================================================================

// Change these values to process different samples
sampleFolder = "sample1";  // Options: "sample1", "sample2", "sample3"
imageNumber = "3";         // Options: "1", "2", "3", "4", etc.

// Auto-generated path
basePath = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10";
dir = basePath + "/" + sampleFolder + "/" + imageNumber + "/";


// ============================================================================
// SETUP - Scale and ROI Manager
// ============================================================================
print("Setting scale (40x objective, 5 pixels per micron)...");
run("Set Scale...", "distance=5 known=1 unit=um global");

print("Opening ROI Manager...");
run("ROI Manager...");


// ============================================================================
// ACTIN-FITC CHANNEL
// ============================================================================
print("\n--- Processing Actin-FITC ---");

// Configure measurements for Actin-FITC
run("Set Measurements...", "area mean standard min centroid shape integrated display redirect=Actin-FITC.tif decimal=3");

// Measure all ROIs
roiManager("Select All");
roiManager("Measure");

// Save and clear
print("Saving Actin-FITC measurements...");
saveAs("Results", dir + "actin-fitc-measurements.csv");
run("Clear Results");


// ============================================================================
// CD4-PerCP CHANNEL
// ============================================================================
print("\n--- Processing CD4-PerCP ---");

// Open and preprocess
open(dir + "CD4-PerCP.tif");
run("Subtract Background...", "rolling=100 sliding");

// Configure measurements
run("Set Measurements...", "area mean standard min centroid shape integrated display redirect=CD4-PerCP.tif decimal=3");

// Measure all ROIs
roiManager("Select All");
roiManager("Measure");

// Save and clear
print("Saving CD4-PerCP measurements...");
saveAs("Results", dir + "cd4-percp-measurements.csv");
run("Clear Results");

// Close image
selectWindow("CD4-PerCP.tif");
close();


// ============================================================================
// CD45RA-AF647 CHANNEL
// ============================================================================
print("\n--- Processing CD45RA-AF647 ---");

// Open channel
open(dir + "CD45RA-AF647.tif");

// Configure measurements
run("Set Measurements...", "area mean standard min centroid shape integrated display redirect=CD45RA-AF647.tif decimal=3");

// Measure all ROIs
roiManager("Select All");
roiManager("Measure");

// Save and clear
print("Saving CD45RA-AF647 measurements...");
saveAs("Results", dir + "cd45ra-af647-measurements.csv");
run("Clear Results");

// Close image
selectWindow("CD45RA-AF647.tif");
close();


// ============================================================================
// CCR7-PE CHANNEL
// ============================================================================
print("\n--- Processing CCR7-PE ---");

// Open and preprocess
open(dir + "CCR7-PE.tif");
run("Subtract Background...", "rolling=50");

// Configure measurements
run("Set Measurements...", "area mean standard min centroid shape integrated display redirect=CCR7-PE.tif decimal=3");

// Measure all ROIs
roiManager("Select All");
roiManager("Measure");

// Save
print("Saving CCR7-PE measurements...");
saveAs("Results", dir + "ccr7-pe-measurements.csv");


// ============================================================================
// COMPLETE
// ============================================================================
print("\n✅ All measurements completed and saved!");

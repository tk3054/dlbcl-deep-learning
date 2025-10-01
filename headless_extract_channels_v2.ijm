// ImageJ Macro: Extract Channel Measurements (HEADLESS VERSION - No ROI Manager)
// Measures fluorescence intensity in all channels without using ROI Manager

// ============================================================================
// CONFIGURATION - EDIT THESE
// ============================================================================

// Change these values to process different samples
sampleFolder = "sample1";  // Options: "sample1", "sample2", "sample3"
imageNumber = "4";         // Options: "1", "2", "3", "4", etc.

// Auto-generated path
basePath = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10";
dir = basePath + "/" + sampleFolder + "/" + imageNumber + "/";


// ============================================================================
// SETUP
// ============================================================================

// Enable batch mode for headless operation
setBatchMode(true);

print("Setting scale (40x objective, 5 pixels per micron)...");
run("Set Scale...", "distance=5 known=1 unit=um global");

// Get list of ROI files
roi_folder = dir + "cell_rois/";
roi_list = getFileList(roi_folder);

// Filter to only .tif files and sort
roi_files = newArray(0);
for (i = 0; i < roi_list.length; i++) {
    if (endsWith(roi_list[i], ".tif")) {
        roi_files = Array.concat(roi_files, roi_list[i]);
    }
}
Array.sort(roi_files);

print("Found " + roi_files.length + " ROI masks");


// ============================================================================
// HELPER FUNCTION: Measure channel with ROI masks
// ============================================================================

function measureChannel(channel_name, channel_file, roi_files, roi_folder, output_file, preprocess) {
    print("\n--- Processing " + channel_name + " ---");

    // Open channel image
    if (!File.exists(dir + channel_file)) {
        print("WARNING: " + channel_file + " not found, skipping");
        return;
    }

    open(dir + channel_file);
    channel_id = getImageID();

    // Preprocess if needed
    if (preprocess == "cd4") {
        run("Subtract Background...", "rolling=100 sliding");
    } else if (preprocess == "cd45ra" || preprocess == "ccr7") {
        run("Subtract Background...", "rolling=50");
    }

    // Configure measurements (don't redirect to avoid auto-clear)
    run("Set Measurements...", "area mean standard min centroid shape integrated display decimal=3");

    // Make sure results table doesn't auto-clear
    run("Clear Results");

    // Measure each ROI
    for (i = 0; i < roi_files.length; i++) {
        print("  Processing ROI " + (i+1) + "/" + roi_files.length + ": " + roi_files[i]);

        // Open ROI mask
        open(roi_folder + roi_files[i]);
        roi_id = getImageID();

        // Create selection from mask
        run("Create Selection");

        // Switch to channel image and measure
        selectImage(channel_id);
        run("Restore Selection");
        run("Measure");

        print("    Results table now has " + nResults + " rows");

        // Close only the ROI mask
        selectImage(roi_id);
        close();

        // Return to channel image
        selectImage(channel_id);
    }

    // Save results
    print("  Total measurements before save: " + nResults);
    print("Saving " + channel_name + " measurements...");
    saveAs("Results", output_file);
    print("  Saved to: " + output_file);
    run("Clear Results");

    // Close channel image
    selectImage(channel_id);
    close();

    print("✓ " + channel_name + " complete (" + roi_files.length + " cells measured)");
}


// ============================================================================
// MEASURE ALL CHANNELS
// ============================================================================

// Actin-FITC
measureChannel("Actin-FITC", "Actin-FITC.tif", roi_files, roi_folder, dir + "actin-fitc-measurements.csv", "none");

// CD4-PerCP
measureChannel("CD4-PerCP", "CD4-PerCP.tif", roi_files, roi_folder, dir + "cd4-percp-measurements.csv", "cd4");

// CD45RA-AF647
measureChannel("CD45RA-AF647", "CD45RA-AF647.tif", roi_files, roi_folder, dir + "cd45ra-af647-measurements.csv", "cd45ra");

// CCR7-PE
measureChannel("CCR7-PE", "CCR7-PE.tif", roi_files, roi_folder, dir + "ccr7-pe-measurements.csv", "ccr7");


// ============================================================================
// COMPLETE
// ============================================================================
print("\n✅ All measurements completed and saved!");

// Disable batch mode
setBatchMode(false);

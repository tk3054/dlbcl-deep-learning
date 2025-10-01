// ImageJ Macro: Extract Channel Measurements (HEADLESS VERSION - Returns Data)
// Measures fluorescence intensity in all channels and returns structured data

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

numCells = roi_files.length;
print("Found " + numCells + " ROI masks");


// ============================================================================
// HELPER FUNCTION: Measure channel with ROI masks
// ============================================================================

function measureChannel(channel_name, channel_file, roi_files, roi_folder, preprocess) {
    print("\n--- Processing " + channel_name + " ---");

    // Open channel image
    if (!File.exists(dir + channel_file)) {
        print("WARNING: " + channel_file + " not found, skipping");
        return "SKIP";
    }

    open(dir + channel_file);
    channel_id = getImageID();

    // Set scale (40x objective, 5 pixels per micron)
    run("Set Scale...", "distance=5 known=1 unit=um");

    // Preprocess if needed
    if (preprocess == "cd4") {
        run("Subtract Background...", "rolling=100 sliding");
    } else if (preprocess == "cd45ra" || preprocess == "ccr7") {
        run("Subtract Background...", "rolling=50");
    }

    // Prepare output string with header
    output = "cell_id,area,mean,std,min,max,x,y,circ,intden,rawintden,ar,round,solidity\n";

    // Measure each ROI
    for (i = 0; i < roi_files.length; i++) {
        print("  Processing ROI " + (i+1) + "/" + roi_files.length + ": " + roi_files[i]);

        // Open ROI mask
        open(roi_folder + roi_files[i]);
        roi_id = getImageID();
        print("    Opened ROI mask, ID: " + roi_id);

        // Create selection from mask
        run("Create Selection");

        // Check if selection was created
        if (selectionType() == -1) {
            print("    ⚠️  WARNING: No selection created from mask (empty ROI?)");
            close();
            continue;
        }
        print("    Created selection");

        // Switch to channel image
        selectImage(channel_id);
        run("Restore Selection");
        print("    Restored selection on channel image");

        // Get measurements manually
        run("Set Measurements...", "area mean standard min centroid shape integrated redirect=None decimal=3");
        run("Measure");
        print("    Measured, nResults = " + nResults);

        // Extract values from Results table (last row)
        row = nResults - 1;
        area = getResult("Area", row);
        mean = getResult("Mean", row);
        std = getResult("StdDev", row);
        min = getResult("Min", row);
        max = getResult("Max", row);
        x = getResult("X", row);
        y = getResult("Y", row);
        circ = getResult("Circ.", row);
        intden = getResult("IntDen", row);
        rawintden = getResult("RawIntDen", row);
        ar = getResult("AR", row);
        roundness = getResult("Round", row);
        solidity = getResult("Solidity", row);

        // Build CSV row
        cell_id = i + 1;
        output = output + cell_id + "," + area + "," + mean + "," + std + "," + min + "," + max + ",";
        output = output + x + "," + y + "," + circ + "," + intden + "," + rawintden + ",";
        output = output + ar + "," + roundness + "," + solidity + "\n";
        print("    ✓ Cell " + cell_id + ": area=" + area + ", mean=" + mean);

        // Clear results for next measurement
        run("Clear Results");

        // Close only the ROI mask
        selectImage(roi_id);
        close();
        print("    Closed ROI mask");

        // Return to channel image
        selectImage(channel_id);
    }

    // Close channel image
    selectImage(channel_id);
    close();

    print("✓ " + channel_name + " complete (" + roi_files.length + " cells measured)");

    return output;
}


// ============================================================================
// MEASURE ALL CHANNELS AND BUILD OUTPUT
// ============================================================================

// Measure each channel
actin_data = measureChannel("Actin-FITC", "Actin-FITC.tif", roi_files, roi_folder, "none");
cd4_data = measureChannel("CD4-PerCP", "CD4-PerCP.tif", roi_files, roi_folder, "cd4");
cd45ra_data = measureChannel("CD45RA-AF647", "CD45RA-AF647.tif", roi_files, roi_folder, "cd45ra");
ccr7_data = measureChannel("CCR7-PE", "CCR7-PE.tif", roi_files, roi_folder, "ccr7");

// Save each channel's data to a file
File.saveString(actin_data, dir + "actin-fitc-measurements.csv");
File.saveString(cd4_data, dir + "cd4-percp-measurements.csv");
File.saveString(cd45ra_data, dir + "cd45ra-af647-measurements.csv");
File.saveString(ccr7_data, dir + "ccr7-pe-measurements.csv");

print("\n✅ All measurements completed and saved!");

// Disable batch mode
setBatchMode(false);

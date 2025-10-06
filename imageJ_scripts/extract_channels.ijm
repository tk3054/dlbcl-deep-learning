// ImageJ Macro: Extract Channel Measurements (HEADLESS VERSION - Returns Data)
// Measures fluorescence intensity in all channels and returns structured data

// ============================================================================
// CONFIGURATION - Set by Python (do not edit manually)
// ============================================================================

// These values are replaced by Python when running the pipeline
sampleFolder = "SAMPLE_PLACEHOLDER";
imageNumber = "IMAGE_PLACEHOLDER";
basePath = "BASE_PATH_PLACEHOLDER";
dir = basePath + "/" + sampleFolder + "/" + imageNumber + "/";

// Channel filenames - replaced by Python from CHANNEL_CONFIG
actinFile = "ACTIN_FILE_PLACEHOLDER";
cd4File = "CD4_FILE_PLACEHOLDER";
cd45raAF647File = "CD45RA_AF647_FILE_PLACEHOLDER";
cd45raSparkVioletFile = "CD45RA_SPARKVIOLET_FILE_PLACEHOLDER";
cd19carFile = "CD19CAR_FILE_PLACEHOLDER";
ccr7File = "CCR7_FILE_PLACEHOLDER";


// ============================================================================
// SETUP
// ============================================================================

// Enable batch mode for headless operation
setBatchMode(true);

// Suppress Results display
setBatchMode("hide");

// Get list of ROI files (using shrunk ROIs for cleaner measurements)
roi_folder = dir + "cell_rois_shrunk/";
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


// ============================================================================
// HELPER FUNCTION: Measure channel with ROI masks
// ============================================================================

function measureChannel(channel_name, channel_file, roi_files, roi_folder, preprocess) {
    // Try to find processed file first, fall back to original
    processed_file = "processed_" + channel_file;
    file_to_use = "";

    if (File.exists(dir + processed_file)) {
        file_to_use = processed_file;
        print("  Using preprocessed file: " + processed_file);
    } else if (File.exists(dir + channel_file)) {
        file_to_use = channel_file;
        print("  Using original file: " + channel_file);
    } else {
        print("WARNING: Neither " + processed_file + " nor " + channel_file + " found, skipping");
        return "SKIP";
    }

    open(dir + file_to_use);
    channel_id = getImageID();

    // Set scale (40x objective, 5 pixels per micron)
    run("Set Scale...", "distance=5 known=1 unit=um");

    // Preprocessing is now done in separate step, so skip here
    // (kept for backward compatibility if processed files don't exist)

    // Prepare output string with header
    output = "cell_id,area,mean,median,std,min,max,x,y,circ,intden,rawintden,ar,round,solidity\n";

    // Measure each ROI
    for (i = 0; i < roi_files.length; i++) {
        // Open ROI mask
        open(roi_folder + roi_files[i]);
        roi_id = getImageID();

        // Create selection from mask
        run("Create Selection");

        // Check if selection was created
        if (selectionType() == -1) {
            print("    ⚠️  WARNING: No selection created from mask in " + roi_files[i]);
            close();
            continue;
        }

        // Switch to channel image
        selectImage(channel_id);
        run("Restore Selection");

        // Get measurements manually
        run("Set Measurements...", "area mean standard min median centroid shape integrated redirect=None decimal=3");
        run("Measure");

        // Extract values from Results table (last row)
        row = nResults - 1;
        area = getResult("Area", row);
        mean = getResult("Mean", row);
        median = getResult("Median", row);
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
        output = output + cell_id + "," + area + "," + mean + "," + median + "," + std + "," + min + "," + max + ",";
        output = output + x + "," + y + "," + circ + "," + intden + "," + rawintden + ",";
        output = output + ar + "," + roundness + "," + solidity + "\n";

        // Clear results for next measurement
        run("Clear Results");

        // Close only the ROI mask
        selectImage(roi_id);
        close();

        // Return to channel image
        selectImage(channel_id);
    }

    // Close channel image
    selectImage(channel_id);
    close();

    return output;
}


// ============================================================================
// MEASURE ALL CHANNELS AND BUILD OUTPUT
// ============================================================================

// Measure each channel (using configured filenames)
actin_data = measureChannel("Actin-FITC", actinFile, roi_files, roi_folder, "none");
cd4_data = measureChannel("CD4-PerCP", cd4File, roi_files, roi_folder, "cd4");
cd45ra_af647_data = measureChannel("CD45RA-AF647", cd45raAF647File, roi_files, roi_folder, "cd45ra");
cd45ra_sparkviolet_data = measureChannel("CD45RA-SparkViolet", cd45raSparkVioletFile, roi_files, roi_folder, "cd45ra");
cd19car_data = measureChannel("CD19CAR-AF647", cd19carFile, roi_files, roi_folder, "none");
ccr7_data = measureChannel("CCR7-PE", ccr7File, roi_files, roi_folder, "ccr7");

// Save each channel's data to a file
File.saveString(actin_data, dir + "actin-fitc-measurements.csv");
File.saveString(cd4_data, dir + "cd4-percp-measurements.csv");
File.saveString(cd45ra_af647_data, dir + "cd45ra-af647-measurements.csv");
File.saveString(cd45ra_sparkviolet_data, dir + "cd45ra-sparkviolet-measurements.csv");
File.saveString(cd19car_data, dir + "cd19car-af647-measurements.csv");
File.saveString(ccr7_data, dir + "ccr7-pe-measurements.csv");

// Disable batch mode
setBatchMode(false);

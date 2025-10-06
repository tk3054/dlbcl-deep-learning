// ImageJ Macro: Preprocess Channels
// Applies channel-specific preprocessing and saves processed TIF files
//
// FUNCTIONALITY:
// - Detects all channel TIF files automatically
// - Applies appropriate preprocessing for each channel:
//   * CCR7-PE: Background subtraction (100px) + Gaussian blur (sigma=8)
//   * CD45RA: Background subtraction (500px) + Gaussian blur (sigma=8)
//   * CD4-PerCP: Background subtraction (100px sliding)
//   * Others: Copy as-is (no preprocessing)
// - Saves as processed_<channelname>.tif
//
// Author: Automated pipeline

// ============================================================================
// CONFIGURATION - Set by Python (do not edit manually)
// ============================================================================

// These values are replaced by Python when running the pipeline
sampleFolder = "SAMPLE_PLACEHOLDER";
imageNumber = "IMAGE_PLACEHOLDER";
basePath = "BASE_PATH_PLACEHOLDER";

// Auto-generated path
dir = basePath + "/" + sampleFolder + "/" + imageNumber + "/";


// ============================================================================
// HELPER FUNCTIONS (inline - no external dependencies)
// ============================================================================

/**
 * Detect all channel TIF files in directory
 * Returns array of original channel TIF filenames
 */
function detectAllChannels(directory) {
    fileList = getFileList(directory);
    channels = newArray(0);

    for (i = 0; i < fileList.length; i++) {
        fileName = fileList[i];

        // Only process .tif files
        if (!endsWith(fileName, ".tif") && !endsWith(fileName, ".TIF")) {
            continue;
        }

        // Skip if it's a processed file
        if (indexOf(fileName, "processed_") >= 0 ||
            indexOf(fileName, "_raw") >= 0 ||
            indexOf(fileName, "_mask") >= 0 ||
            indexOf(fileName, "original_image") >= 0 ||
            indexOf(fileName, "duplicate") >= 0 ||
            indexOf(fileName, "_crop") >= 0 ||
            startsWith(fileName, "cell_")) {
            continue;
        }

        // This is an original channel file
        channels = Array.concat(channels, fileName);
    }

    return channels;
}

/**
 * Get channel name from filename
 * Examples: "CD19-APC.tif" -> "CD19-APC", "Actin-FITC.tif" -> "Actin-FITC"
 */
function getChannelNameFromFile(fileName) {
    // Remove .tif extension
    baseName = replace(fileName, ".tif", "");
    baseName = replace(baseName, ".TIF", "");

    return baseName;
}

/**
 * Determine preprocessing type based on channel name
 */
function getPreprocessType(channelName) {
    upperName = toUpperCase(channelName);

    if (indexOf(upperName, "CCR7") >= 0) {
        return "ccr7";
    } else if (indexOf(upperName, "CD45RA") >= 0) {
        return "cd45ra";
    } else if (indexOf(upperName, "CD4") >= 0) {
        return "cd4";
    } else {
        return "none";
    }
}


// ============================================================================
// DETECT ALL CHANNELS
// ============================================================================

// Enable batch mode for headless operation
setBatchMode(true);

// Detect all channel files in directory
print("Detecting channel files in: " + dir);
allChannels = detectAllChannels(dir);

if (allChannels.length == 0) {
    print("ERROR: No channel TIF files found in directory!");
    exit("No channel TIF files found!");
}

print("Found " + allChannels.length + " channel files:");
for (i = 0; i < allChannels.length; i++) {
    print("  - " + allChannels[i]);
}


// ============================================================================
// MAIN PROCESSING - Preprocess and save all channels
// ============================================================================

print("\nPreprocessing channels...");
for (i = 0; i < allChannels.length; i++) {
    channelFile = allChannels[i];
    channelName = getChannelNameFromFile(channelFile);
    preprocessType = getPreprocessType(channelName);

    print("  Processing channel: " + channelName + " (type: " + preprocessType + ")");

    // Open the channel file
    open(dir + channelFile);
    original_id = getImageID();

    // Apply preprocessing based on channel type
    if (preprocessType == "ccr7") {
        print("    Applying: Background subtraction (50px) + Gaussian blur (sigma=8)");
        run("Subtract Background...", "rolling=100");
        run("Gaussian Blur...", "sigma=8");
    } else if (preprocessType == "cd45ra") {
        print("    Applying: Background subtraction (500px) + Gaussian blur (sigma=8)");
        run("Subtract Background...", "rolling=500");
        run("Gaussian Blur...", "sigma=8");
    } else if (preprocessType == "cd4") {
        print("    Applying: Background subtraction (100px sliding)");
        run("Subtract Background...", "rolling=100 sliding");
    } else {
        print("    No preprocessing needed, copying as-is");
    }

    // Save processed file
    processedFilename = "processed_" + channelName + ".tif";
    fullPath = dir + processedFilename;
    saveAs("Tiff", fullPath);
    print("    Saved: " + processedFilename);

    // Close the image
    close();
}

// Disable batch mode
setBatchMode(false);

print("\nPreprocessing complete!");
print("Created " + allChannels.length + " processed channel files");

// ImageJ Macro: Create JPG Duplicates of All Channel TIF Files
// Creates JPG duplicates for all channel TIF files in a directory
//
// FUNCTIONALITY:
// - Detects all channel TIF files automatically
// - Creates JPG duplicate for each channel with naming: *_<channelname>_raw.jpg
// - Can be run from Python automatically
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

// File naming configuration
donorName = "CLLSaSa";
date = "07292025";
stiffness = "1to10";
spreadingTime = "40min";
sampleNumber = "1";


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
        if (indexOf(fileName, "_raw") >= 0 ||
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
 * Format image number with zero padding
 * Examples: "2" -> "02", "2[CAR]" -> "02[CAR]", "10" -> "10"
 */
function formatImageNumber(imageNumber) {
    // Check if has bracket suffix like [CAR]
    bracketIndex = indexOf(imageNumber, "[");

    if (bracketIndex > 0) {
        // Has suffix - split it
        numPart = substring(imageNumber, 0, bracketIndex);
        suffixPart = substring(imageNumber, bracketIndex);

        // Pad number part if single digit
        if (lengthOf(numPart) == 1) {
            return "0" + numPart + suffixPart;
        } else {
            return imageNumber;
        }
    } else {
        // No suffix - just pad if single digit
        if (lengthOf(imageNumber) == 1) {
            return "0" + imageNumber;
        } else {
            return imageNumber;
        }
    }
}

/**
 * Generate output filename with proper formatting
 */
function generateOutputFilename(imgNumber, suffix, extension, donor, dt, stiff, spread, sample) {
    // Format image number with zero padding while preserving suffix like [CAR]
    imageNumberFormatted = formatImageNumber(imgNumber);

    // Construct filename
    filename = donor + "_" + dt + "_" + stiff + "_" + spread + "_" + sample + "_" + imageNumberFormatted + suffix + extension;

    return filename;
}


// ============================================================================
// HELPER FUNCTION: Find processed version of file
// ============================================================================

function findProcessedVersion(fileName) {
    // Try to find processed_ version
    channelName = getChannelNameFromFile(fileName);
    processedFile = "processed_" + channelName + ".tif";

    if (File.exists(dir + processedFile)) {
        return processedFile;
    } else {
        return fileName;  // Fall back to original
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
// MAIN PROCESSING - Create JPG duplicates for ALL channels
// ============================================================================

print("\nImage number: " + imageNumber);
print("Image directory: " + dir);

print("\nCreating JPG duplicates for all channels...");
for (i = 0; i < allChannels.length; i++) {
    channelFile = allChannels[i];
    channelName = getChannelNameFromFile(channelFile);

    // Try to use processed version if available
    fileToUse = findProcessedVersion(channelFile);

    if (fileToUse != channelFile) {
        print("  Processing channel: " + channelName + " (using processed version)");
    } else {
        print("  Processing channel: " + channelName);
    }

    // Open the channel file (processed or original)
    open(dir + fileToUse);
    openedWindow = getTitle();

    // Select and duplicate
    selectWindow(openedWindow);
    run("Duplicate...", "title=raw_temp_" + i);

    // Generate filename: use channel name as suffix
    // Example: CLLSaSa_07292025_1to10_40min_1_01[CAR]_Actin-FITC_raw.jpg
    rawFilename = generateOutputFilename(imageNumber, "_" + channelName + "_raw", ".jpg", donorName, date, stiffness, spreadingTime, sampleNumber);
    fullRawPath = dir + rawFilename;
    saveAs("Jpeg", fullRawPath);
    print("    Saved: " + rawFilename);
    close(); // Close duplicate

    // Close original
    selectWindow(openedWindow);
    close();
}

// Disable batch mode
setBatchMode(false);

print("\nProcessing complete!");
print("Created " + allChannels.length + " JPG duplicate files");

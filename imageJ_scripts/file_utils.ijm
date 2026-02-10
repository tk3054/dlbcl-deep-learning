// ============================================================================
// ImageJ File Finding Utilities
// Helper functions to intelligently find files in image directories
// ============================================================================

/**
 * Find raw image file in directory
 * Returns the filename (not full path) or empty string if not found
 */
function findRawImage(dir) {
    fileList = getFileList(dir);

    // Look for files ending in _raw.jpg
    for (i = 0; i < fileList.length; i++) {
        if (endsWith(fileList[i], "_raw.jpg")) {
            return fileList[i];
        }
    }

    return "";  // Not found
}

/**
 * Find original Actin-FITC image
 * Returns the filename or empty string if not found
 */
function findActinImage(dir) {
    fileList = getFileList(dir);

    // Priority order
    candidates = newArray("Actin-FITC.tif", "actin-fitc.tif", "Actin.tif");

    for (i = 0; i < candidates.length; i++) {
        for (j = 0; j < fileList.length; j++) {
            if (fileList[j] == candidates[i]) {
                return fileList[j];
            }
        }
    }

    return "";  // Not found
}

/**
 * Find channel file by pattern
 * channelName: e.g., "CD19", "CCR7", "CD3"
 * Returns filename or empty string
 */
function findChannelFile(dir, channelName) {
    fileList = getFileList(dir);

    // Look for files containing the channel name
    for (i = 0; i < fileList.length; i++) {
        fileName = fileList[i];
        upperFileName = toUpperCase(fileName);
        upperChannelName = toUpperCase(channelName);

        if (indexOf(upperFileName, upperChannelName) >= 0 && endsWith(fileName, ".tif")) {
            return fileName;
        }
    }

    return "";  // Not found
}

/**
 * Detect all channel files in directory
 * Returns array of original channel TIF filenames
 * Simply finds all .tif files that aren't processed outputs
 */
function detectAllChannels(dir) {
    fileList = getFileList(dir);
    channels = newArray(0);

    for (i = 0; i < fileList.length; i++) {
        fileName = fileList[i];

        // Only process .tif files
        if (!endsWith(fileName, ".tif") && !endsWith(fileName, ".TIF")) {
            continue;
        }

        // Skip if it's a processed file (_raw, _mask, original_image, duplicate, etc.)
        if (indexOf(fileName, "_raw") >= 0 ||
            indexOf(fileName, "_mask") >= 0 ||
            indexOf(fileName, "original_image") >= 0 ||
            startsWith(fileName, "duplicate-")) {
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
 * List all image files in directory
 * Returns array of image filenames
 */
function listAllImages(dir) {
    fileList = getFileList(dir);
    imageExts = newArray(".tif", ".tiff", ".jpg", ".jpeg", ".png");

    images = newArray(0);

    for (i = 0; i < fileList.length; i++) {
        fileName = fileList[i];
        isImage = false;

        for (j = 0; j < imageExts.length; j++) {
            if (endsWith(toLowerCase(fileName), imageExts[j])) {
                isImage = true;
                break;
            }
        }

        if (isImage) {
            images = Array.concat(images, fileName);
        }
    }

    return images;
}

/**
 * Generate output filename with proper formatting
 * imageNumber: e.g., "2[CAR]", "5", "10[CAR]"
 * suffix: e.g., "_raw", "_mask"
 * extension: e.g., ".jpg", ".tif"
 * Returns formatted filename
 */
function generateOutputFilename(imageNumber, suffix, extension, donorName, date, stiffness, spreadingTime, sampleNumber) {
    // Format image number with zero padding while preserving suffix like [CAR]
    imageNumberFormatted = formatImageNumber(imageNumber);

    // Construct filename
    filename = donorName + "_" + date + "_" + stiffness + "_" + spreadingTime + "_" + sampleNumber + "_" + imageNumberFormatted + suffix + extension;

    return filename;
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
 * Print all images found in directory (for debugging)
 */
function printAllImagesInDir(dir) {
    print("Files in directory: " + dir);
    images = listAllImages(dir);

    if (images.length == 0) {
        print("  No image files found");
    } else {
        for (i = 0; i < images.length; i++) {
            print("  - " + images[i]);
        }
    }
}

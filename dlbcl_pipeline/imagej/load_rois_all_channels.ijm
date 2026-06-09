// ============================================================================
// CONFIGURATION - EDIT THESE
// ============================================================================

// Change these values to process different samples
sampleFolder = "sample1";  // Options: "sample1", "sample2", etc.
imageNumber = "1[CAR]";         // Options: "1", "2", "3", "4", "5", etc.

// Auto-generated paths
basePath = "/Users/taeeonkong/Desktop/2025 Fall Images/09-26-2025 DLBCL";
baseDir = basePath + "/" + sampleFolder + "/" + imageNumber + "/";
roi_folder = baseDir + "cell_rois/";


// ============================================================================
// LOAD ROIs
// ============================================================================

// Load All Cell ROIs onto All Open Images

// Get list of currently open images
imageIDs = newArray(nImages);
for (i = 0; i < nImages; i++) {
    selectImage(i + 1);
    imageIDs[i] = getImageID();
}

// Get list of ROI files
list = getFileList(roi_folder);

// Load each ROI into ROI Manager
for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tif")) {
        // Open ROI mask
        open(roi_folder + list[i]);

        // Convert to 8-bit if needed
        if (bitDepth() != 8) {
            run("8-bit");
        }

        // Threshold to create binary mask
        setThreshold(1, 255);
        run("Convert to Mask");

        // Convert to selection and add to ROI Manager
        run("Create Selection");
        roiManager("Add");
        roiManager("Select", roiManager("count")-1);
        roiManager("Rename", "Cell_" + (i+1));

        // Close ROI mask
        close();
    }
}

print("Loaded " + roiManager("count") + " ROIs");

// Show ROIs on each open image
for (i = 0; i < imageIDs.length; i++) {
    selectImage(imageIDs[i]);
    roiManager("Show All");
}
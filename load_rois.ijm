// ============================================================================
// CONFIGURATION - EDIT THESE
// ============================================================================

// Change these values to process different samples
sampleFolder = "sample1";  // Options: "sample1", "sample2", "sample3"
imageNumber = "3";         // Options: "1", "2", "3", "4", etc.

// Auto-generated paths
basePath = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10";
baseDir = basePath + "/" + sampleFolder + "/" + imageNumber + "/";
roi_folder = baseDir + "cell_rois/";
original_image = baseDir + "original_image.tif";


// ============================================================================
// LOAD ROIs
// ============================================================================

// Load All Cell ROIs onto Original Image

// Open original image
open(original_image);
original_id = getImageID();

  // Get list of ROI files
  list = getFileList(roi_folder);

  // Load each ROI
  for (i = 0; i < list.length; i++) {
      if (endsWith(list[i], ".tif")) {
          // Open ROI mask
          open(roi_folder + list[i]);

          // Convert to selection and add to ROI Manager
          run("Create Selection");
          roiManager("Add");
          roiManager("Select", roiManager("count")-1);
          roiManager("Rename", "Cell_" + (i+1));

          // Close ROI mask
          close();

          // Make sure original image is still active
          selectImage(original_id);
      }
  }

  print("Loaded " + roiManager("count") + " ROIs");
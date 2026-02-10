  // Extract Individual Cells as Square Images (like Python script)
  output_folder = "/Users/taeeonkong/Desktop/Project/Summer2025/20250729_CLLSaSa/1to10/sample1/1/extracted_cells/";
  File.makeDirectory(output_folder);

  target_size = 64; // Same as your Python script

  // Get the current image
  original_id = getImageID();
  original_title = getTitle();

  // Check if ROI Manager has ROIs
  if (roiManager("count") == 0) {
      exit("No ROIs found! Please load ROIs first.");
  }

  // Extract each cell
  for (i = 0; i < roiManager("count"); i++) {
      // Select original image
      selectImage(original_id);

      // Select current ROI
      roiManager("Select", i);

      // Get ROI name
      roi_name = call("ij.plugin.frame.RoiManager.getName", i);
      if (roi_name == "") roi_name = "Cell_" + (i+1);

      // Duplicate the selected area (crop)
      run("Duplicate...", "title=temp_crop");
      crop_id = getImageID();

      // Get current dimensions
      w = getWidth();
      h = getHeight();

      // Skip if empty
      if (w == 0 || h == 0) {
          close();
          continue;
      }

      // Calculate padding needed for square
      max_dim = Math.max(w, h);
      target_dim = Math.max(max_dim, target_size);

      // Add padding to make square, centered (like your Python script)
      run("Canvas Size...",
          "width=" + target_dim + " height=" + target_dim + " position=Center zero");

      // Resize to target size if needed (preserving aspect ratio)
      if (target_dim > target_size) {
          run("Size...", "width=" + target_size + " height=" + target_size + " interpolation=Bilinear");
      }

      // Save as JPG with high quality
      saveAs("Jpeg", output_folder + roi_name + "_" + original_title + ".jpg");
      close();
  }

  print("Extracted " + roiManager("count") + " square cells (" + target_size + "x" + target_size + ") to: " + output_folder);
# Digital Image Forgery Detection and Localization

CSE4007 – Digital Image Processing (IA-3)

This project finds **possible copy-move forgery**: content copied from one place
to another inside the same image. It uses DCT, ORB, geometric verification and
morphology. There is no AI model or training step.

Both MATLAB and C++ implementations save the input, intermediate masks, matching
lines, highlighted regions and a text report. The project stays focused on a
single-image detector; a web app or database is not needed.

## Run the C++ version

Requirements: CMake 3.16+, a C++17 compiler, and OpenCV 4 development files.
On Ubuntu/Debian, install missing dependencies with:

```bash
sudo apt install build-essential cmake libopencv-dev
```

From the project folder:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/forgery_detector input/example.png results/cpp/example
```

Put new photos in `input/`. The truth mask is optional:

```bash
./build/forgery_detector input/my_photo.jpg results/cpp/my_photo
```

The arguments are `input_image [output_folder] [truth_mask]`. Outputs default to
`results` if the folder is omitted. Give each image a separate output folder;
running again in the same folder replaces that run's results. The program saves
images rather than opening a desktop window, so it also runs on a terminal-only
machine. On Windows, use the generated `forgery_detector.exe` path.

## Run the MATLAB version

Requirements: **MATLAB R2022b or later**, **Image Processing Toolbox**, and
**Computer Vision Toolbox**. The geometric verification function
[`estgeotform2d`](https://www.mathworks.com/help/vision/ref/estgeotform2d.html)
is available from R2022b.

Put a photo in `input/`, then set MATLAB's current folder to this project:

```matlab
addpath('matlab');
result = evaluation('input/example.png');
```

This saves results under `results/matlab/example` and opens a six-panel figure.

```matlab
% Analyze your own image without a truth mask.
result = evaluation('path/to/photo.png');

% Evaluate a known mask; false turns off the figure.
result = evaluation('path/to/photo.png', 'path/to/mask.png', 'results/my_photo', false);

% Recreate the three synthetic demonstration images if needed.
create_demo;

% Run basic functional checks, reporting unavailable toolbox stages as skipped.
test_project;
```

`evaluation` returns counts, boxes, the final logical mask, time and optional
evaluation scores. It also saves `result.mat`. It reports a missing toolbox
clearly instead of silently skipping ORB. Octave can generate the demo images,
but is not a replacement for MATLAB's Computer Vision Toolbox.

## What each stage does

1. **Preprocessing:** read the image, convert to 8-bit grayscale and limit its
   longest side to 768 pixels. Keep a color copy for the overlay.
2. **DCT:** inspect overlapping 16×16 blocks at a stride of 4. Skip flat blocks,
   keep 15 low-frequency AC coefficients and normalize them. Compare similar
   descriptors and reject matches less than 32 pixels apart.
3. **Block verification:** group pairs by their displacement. Require at least
   12 distinct pairs forming a connected patch, with enough spread at both ends.
4. **ORB:** compare binary corner descriptors using Hamming distance. Reject
   nearby, ambiguous and duplicate matches, then check a common similarity
   transformation with RANSAC (MATLAB uses its MSAC variant). Require at least
   eight spatially distributed inliers. This can help with some rotated or
   resized copies.
5. **Localization:** combine the verified masks, close small gaps, open thin
   noise and remove connected regions smaller than 180 pixels. Highlight the
   remaining regions in red and draw yellow boxes.
6. **Evaluation:** when a truth mask is provided, calculate precision, recall,
   F1 and intersection over union (IoU) from actual detected pixels.

The thresholds are kept near the code that uses them. MATLAB searches a short
window of sorted DCT descriptors; C++ uses FLANN nearest neighbors. MATLAB keeps
up to 900 ORB points; C++ requests 2,500. Their detectors, interpolation and
morphology implementations differ, so their outputs need not be identical.

## Saved outputs

| File | Contents |
| --- | --- |
| `01_input.png` | Color image at processing resolution |
| `02_grayscale.png` | Grayscale input to the detectors |
| `03_dct_mask.png` | Verified duplicated blocks |
| `04_orb_mask.png` | Neighborhoods around verified ORB points |
| `05_final_mask.png` | Cleaned combined binary mask |
| `06_highlighted.png` | Red overlay and yellow bounding boxes |
| `07_orb_matches.png` | Verified feature correspondences |
| `08_dct_matches.png` | Sample of verified block correspondences |
| `report.txt` | Dimensions, match counts, region counts, boxes, time and scores |

White mask pixels are suspicious; black pixels are background. C++ draws matches
between two views of the same image. MATLAB draws them on one image, limiting
the illustration to 60 lines. The C++ DCT picture shows at most 150 lines.
Reported counts include all accepted matches. Connected regions and displacement
groups are measurements, not a count of editing operations.

All masks and boxes use the **processed image size**. Reports include the original
size too. MATLAB boxes use `regionprops` pixel-edge coordinates; C++ uses zero-based
integer coordinates. MATLAB times detection and localization; C++ additionally
includes resizing and grayscale conversion. Neither timing includes saving the
outputs, so these are not directly comparable performance benchmarks.

## Testing with known masks

Image files outside `input/` have been removed from the workspace. Put a photo
you want to inspect in `input/`. For pixel scores, add a grayscale truth mask
with the same dimensions and white pixels over both matching regions. The
detector can run without a mask, but cannot calculate mask scores.

For a larger evaluation, obtain images and masks from the official
[CoMoFoD dataset](https://www.vcl.fer.hr/comofod/index.html) under its release
terms. Use the original photographs as negative controls and forged photographs
as positive cases. Include translation, rotation, resizing, JPEG compression and
blur in your evaluation. The dataset is not currently in the workspace.

Supply a lossless mask with the same dimensions as the **original** input image.
Any nonzero color channel is foreground; alpha is ignored. If the input is resized,
the mask is resized with nearest-neighbor interpolation. A mask labeling only the
pasted destination is not directly comparable with this detector, which marks
both matching regions. Verify the dataset's mask convention first.

`TP`, `FP` and `FN` mean true-positive, false-positive and false-negative pixels:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 TP / (2 TP + FP + FN)
IoU       = TP / (TP + FP + FN)
```

A zero denominator returns **0**, including the case where both masks are empty.
For authentic images, use the image-level flag rather than these foreground
scores. Across a labeled set, false-positive rate is the number of authentic
images flagged divided by the total authentic images tested. The program does
not invent a dataset accuracy or infer that rate from one image.

## Project files

```text
matlab/
  preprocessing.m          Image loading and resizing
  dct_block_matching.m     Block features and displacement verification
  orb_matching.m           Feature matching and geometric verification
  region_localization.m    Morphology, overlay and boxes
  evaluation.m             Run the pipeline and save results
  create_demo.m            Reproduce the included synthetic images
  test_project.m           Basic functional checks
cpp/
  Detector.hpp             Shared function declarations and result structures
  Preprocessor.cpp         Image preparation
  DCTDetector.cpp          Block matching
  FeatureMatcher.cpp       ORB matching
  GeometricVerifier.cpp    RANSAC verification
  MaskGenerator.cpp        Mask cleanup and overlay
  main.cpp                 Command-line runner and evaluation report
  CMakeLists.txt           Build configuration
dataset/                   Included demo; place other images here if desired
results/                   Generated output folders
```

## Limits to explain during the presentation

- This detects copy-move indicators, not all types of image manipulation.
- It can mark both matching regions but cannot decide which was the original.
- Natural repeated textures can cause false positives. Smooth or very small
  copied regions may provide too little evidence.
- DCT is strongest for translation. The stride and resizing can miss copies
  whose pixels no longer align with the block grid.
- ORB checks one dominant transformation; multiple edits, large distortions,
  strong compression, blur and poorly textured copies can be missed.
- ORB disks and morphological cleanup give approximate regions, not exact object
  outlines. A negative result does not establish that a photo is authentic.

For the demonstration, run the copied example and its original control, open the
highlighted images and reports, then explain how block similarity, geometric
agreement and morphology contribute to the final mask.

## Validation in this workspace

The C++ program built with OpenCV 4.10 and passed copied-image and authentic-
control checks. Synthetic rotation, resizing, JPEG and blur cases also produced
detections. Those image files were later removed from the workspace. Text reports
remain under `results/cpp/`.

MATLAB R2026a passed `test_project` for preprocessing, DCT, localization and
empty/tiny inputs. This machine has Image Processing Toolbox but lacks Computer
Vision Toolbox, so MATLAB ORB and the complete MATLAB runner were **not executed**.
The test reports those stages as skipped. Install the required toolbox before
using `evaluation`; the C++ pipeline is independently runnable now.

# Digital Image Forgery Detection and Localization

## Detailed project and faculty presentation guide

This guide explains the project in clear language, shows how to run it, and
includes the formulas used by the detector and its evaluation.

## 1. Project overview

**Project title:** Digital Image Forgery Detection and Localization Using DCT,
ORB, and Morphological Processing

**Course:** CSE4007 – Digital Image Processing (IA-3)

**Goal:** Look for signs that part of a photograph was copied and pasted into a
different place in that same photograph. The program highlights possible source
and copied areas and saves measurements.

**Methods:** Discrete Cosine Transform (DCT), ORB feature matching, geometric
verification, and morphological mask cleanup. These are classical digital image
processing methods. The project does not train or use an AI model.

## 2. Problem and real-world example

Imagine a photograph of a road after an accident. Someone could copy one damaged
car and paste it elsewhere in the road to make it look like two cars were
damaged. Or someone could copy a patch of road over an object to hide it. The
edited image may look natural, so a person looking at it might not notice.

This program searches for similar image patterns at separate locations. If many
small regions look alike and their positions follow the same movement, the
program marks them as suspicious.

The result is a reason to inspect the photo more closely. It does not prove that
the image was edited. The program cannot tell which of two matching areas came
first. It focuses on copy-move editing inside one photo; it does not detect every
kind of editing, such as an object copied from a different photograph.

## 3. How the whole program works

1. Read the photo and keep a color copy for the final display.
2. Convert a working copy to grayscale and resize large inputs.
3. Use DCT to compare small overlapping blocks.
4. Use ORB to compare distinctive points such as corners.
5. Check whether the matches follow a shared movement or transformation.
6. Combine both masks and clean gaps, specks, and tiny regions.
7. Save the masks, match pictures, highlighted photo, and report.

    Input photo
       ├── DCT: matching block textures and shared displacement ──┐
       └── ORB: matching points and geometric verification ───────┤
                                                                  ↓
                                  Combine masks → clean regions → results

## 4. Step-by-step method and formulas

### 4.1 Read, grayscale, and resize

The program keeps the color photo for the final overlay and makes a grayscale
copy for matching. A standard grayscale conversion is:

$$
Y(x,y) = 0.299R(x,y) + 0.587G(x,y) + 0.114B(x,y)
$$

Here, R, G, and B are the red, green, and blue pixel values, and Y is brightness.
OpenCV reads color channels in BGR order but applies the corresponding weights
to the correct channels.

For an image of height H and width W, the resize scale is:

$$
s = \min\left(1,\frac{768}{\max(H,W)}\right)
$$

The image is resized only when its longest side is larger than 768 pixels. Both
dimensions are multiplied by s, preserving the aspect ratio. The report records
the original and processed dimensions. Masks and boxes use processed-image
coordinates.

### 4.2 DCT block matching

DCT means Discrete Cosine Transform. It describes the brightness patterns inside
a block using frequency values. Put simply, it turns the texture of a small
patch into a short list of numbers.

The program scans the image with overlapping 16 by 16 pixel blocks, moving
4 pixels at a time. Flat blocks are skipped because plain sky or walls do not
provide distinctive patterns.

For image width W and height H, the number of block positions is:

$$
n_x = \left\lfloor\frac{W-16}{4}\right\rfloor+1,\qquad
n_y = \left\lfloor\frac{H-16}{4}\right\rfloor+1
$$

$$
n_{\mathrm{blocks}} = n_x n_y
$$

For a block X(x,y) of size N by N, with N = 16, the two-dimensional DCT is:

$$
F(u,v)=\alpha(u)\alpha(v)
\sum_{x=0}^{N-1}\sum_{y=0}^{N-1} X(x,y)
\cos\left(\frac{\pi(2x+1)u}{2N}\right)
\cos\left(\frac{\pi(2y+1)v}{2N}\right)
$$

The scale factor is:

$$
\alpha(k)=
\begin{cases}
\sqrt{\frac{1}{N}}, & k=0,\\
\sqrt{\frac{2}{N}}, & k>0.
\end{cases}
$$

The DC coefficient F(0,0) mostly represents average brightness. The detector
skips it and keeps 15 low-frequency AC coefficients in zigzag order. Low
frequencies describe broad patterns and tend to be less sensitive to small
pixel changes than fine details.

Let the selected coefficients be the vector c:

$$
\mathbf{c}=[c_1,c_2,\ldots,c_{15}]
$$

The vector is normalized so overall contrast has less effect:

$$
\mathbf{d}=\frac{\mathbf{c}}{\lVert\mathbf{c}\rVert_2},
\qquad
\lVert\mathbf{c}\rVert_2=\sqrt{\sum_{i=1}^{15}c_i^2}
$$

Blocks with very small descriptor energy are skipped. The distance between two
normalized descriptors is:

$$
D(\mathbf{d}_a,\mathbf{d}_b)
=\sqrt{\sum_{i=1}^{15}(d_{a,i}-d_{b,i})^2}
$$

A smaller distance means the blocks are more alike. The implementation accepts
distances below 0.30 and ignores blocks closer than 32 pixels.

If block A is at (x_a,y_a) and block B is at (x_b,y_b), their displacement is:

$$
\Delta=(x_b-x_a,\;y_b-y_a)
$$

Displacements are grouped on an 8-pixel grid:

$$
g=\operatorname{round}\left(\frac{\Delta}{8}\right)
$$

A likely copy should have many similar block pairs with about the same movement.
The program requires at least 12 distinct pairs. They must form a connected
patch and cover at least 20 pixels horizontally and vertically at both ends.
This helps reject scattered coincidences.

### 4.3 ORB feature matching

ORB means Oriented FAST and Rotated BRIEF. It finds distinctive points, often
corners and textured spots, and describes the nearby image pattern with binary
values. ORB can help with some rotated or resized copies.

FAST looks at pixels around a candidate point to decide whether the center
resembles a corner. ORB estimates the point's direction using image-intensity
moments:

$$
m_{pq}=\sum_{x,y}x^p y^q I(x,y)
$$

$$
\theta=\operatorname{atan2}(m_{01},m_{10})
$$

BRIEF compares pairs of pixel intensities around the point. Each comparison
becomes one bit:

$$
b_k=
\begin{cases}
1, & I(T_\theta(a_k)) < I(T_\theta(b_k)),\\
0, & \text{otherwise}.
\end{cases}
$$

Here, a_k and b_k are sample locations and T_theta rotates the locations to
follow the point's direction. Descriptors are compared by Hamming distance:

$$
D_H(\mathbf{a},\mathbf{b})=\sum_{k=1}^{K}[a_k\ne b_k]
$$

Each bracket is 1 if the bits differ and 0 if they are equal. The C++ program
requests up to 2,500 ORB points; MATLAB keeps up to 900. Both reject nearby,
ambiguous, and duplicate matches. C++ uses a nearest-match ratio check. If d_1
is the closest match distance and d_2 the second closest, it keeps a match when:

$$
d_1 < 0.8d_2,\qquad d_1\leq48
$$

A pair must also select each other as best matches and be at least 48 pixels
apart. Matches in each 8-pixel cell are reduced to avoid counting the same small
area many times.

### 4.4 Geometric verification

Two unrelated corners can look similar by chance. The program checks whether
many matches follow one common transformation. The similarity model includes
rotation, scale, and translation:

$$
x'=s(x\cos\theta-y\sin\theta)+t_x
$$

$$
y'=s(x\sin\theta+y\cos\theta)+t_y
$$

Here, s is scale, theta is rotation, and t_x and t_y are horizontal and vertical
movement. C++ uses RANSAC; MATLAB uses the related MSAC method. Both estimate a
transformation and identify matches that agree with it.

The reprojection error for point pair i is:

$$
e_i=\left\lVert\mathbf{q}_i-T(\mathbf{p}_i)\right\rVert_2
$$

A point pair is counted as an inlier when its error is at most 4 pixels. The
program requires at least 8 inliers, a scale between 0.55 and 1.8, and points
spread over an area rather than in a straight line or tiny cluster. The source
and target point groups each need a 20-pixel span and a convex hull area of at
least 400 square pixels.

### 4.5 Combine and clean the masks

DCT marks the accepted 16 by 16 blocks at both matching locations. ORB marks a
disk of radius 13 pixels around each verified point. The masks are combined
with a logical OR:

$$
M_0(x,y)=M_{\mathrm{DCT}}(x,y)\lor M_{\mathrm{ORB}}(x,y)
$$

Morphology cleans the combined mask. Dilation expands foreground pixels;
erosion shrinks them. For foreground set A and structuring shape B:

$$
A\oplus B=\{z:(\hat{B})_z\cap A\ne\varnothing\}
$$

$$
A\ominus B=\{z:B_z\subseteq A\}
$$

Opening removes small specks; closing joins small gaps:

$$
A\circ B=(A\ominus B)\oplus B
$$

$$
A\bullet B=(A\oplus B)\ominus B
$$

The program applies closing, then opening, then removes connected regions
smaller than 180 pixels. MATLAB uses disk-shaped structuring elements; C++ uses
elliptical elements. Each remaining connected component becomes a region with
a bounding box.

At flagged pixels, the output blends the original pixel I with red R_c:

$$
I_{\mathrm{out}}(x,y)=0.55I(x,y)+0.45R_c
\qquad\text{when }M(x,y)=1
$$

The program draws a yellow box around each region. These boxes are approximate
guides, not exact object outlines.

## 5. Evaluation formulas

When a matching truth mask is supplied, compare the prediction with the truth
one pixel at a time:

- **TP (true positive):** both the prediction and truth mark the pixel.
- **FP (false positive):** the prediction marks it, but truth does not.
- **FN (false negative):** truth marks it, but the prediction does not.
- **TN (true negative):** neither marks the pixel.

The report calculates precision, recall, F1, and intersection over union (IoU):

$$
\mathrm{Precision}=\frac{TP}{TP+FP}
$$

$$
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

$$
F_1=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
=\frac{2TP}{2TP+FP+FN}
$$

$$
\mathrm{IoU}=\frac{TP}{TP+FP+FN}
$$

The program returns 0 if a score's denominator is zero. The truth mask should
match the original photo dimensions and mark both the source and pasted area.
If it labels only the destination, scores will not fairly compare against a
prediction that may include both regions.

The flagged percentage is:

$$
\mathrm{FlaggedArea}(\%)=
100\frac{\sum_{x=1}^{W}\sum_{y=1}^{H}M(x,y)}{WH}
$$

Without a truth mask, the program can show suspicious regions and flagged area,
but cannot calculate precision, recall, F1, or IoU. A collection with authentic
photos is also needed to calculate image-level false-positive rate:

$$
\mathrm{FalsePositiveRate}=
\frac{\text{authentic photos flagged}}{\text{authentic photos tested}}
$$

## 6. Why use C++ for a low-latency version?

C++ is compiled into native machine instructions. It has relatively little
runtime overhead and calls OpenCV's native image-processing routines directly.
That makes it a practical choice when each image needs a quick response, many
images must be processed, or the detector needs to be embedded in another
application.

MATLAB is useful for developing, visualizing, and explaining the image
processing steps. MATLAB also uses compiled and optimized routines, so it is
not accurate to say it is simply interpreted or that C++ is always faster. This
project has not run an apples-to-apples speed benchmark. The two versions use
different search methods and feature limits, so their timings cannot support a
fair speed comparison without first matching settings and measuring the same
operations.

## 7. Put new photos in input/

Place a new photo in the project's input folder. Example:

    input/
      my_photo.jpg
      my_photo_mask.png   (optional)

The optional mask should be a lossless grayscale PNG with black background and
white pixels over both matching areas. It must have the same width and height
as the original photo. A mask is only needed for pixel evaluation scores.

## 8. Run the C++ version

Open a terminal in the project root, ~/ImageForgeryDetector. Build once, or
rebuild after changing C++ files:

    cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j2

Requirements are CMake 3.16 or later, a C++17 compiler, and OpenCV 4 development
files. On Ubuntu or Debian, install missing dependencies with:

    sudo apt install build-essential cmake libopencv-dev

Run a photo without a truth mask:

    ./build/forgery_detector input/my_photo.jpg results/cpp/my_photo

Run with a matching truth mask:

    ./build/forgery_detector input/my_photo.jpg results/cpp/my_photo input/my_photo_mask.png

The general command is:

    forgery_detector <input_image> [output_folder] [truth_mask]

The output folder is created automatically. If omitted, results go into
results/. Each run saves its intermediate images, final mask, overlay, match
visualizations, and report.txt in that folder.

## 9. Run the MATLAB version

The complete MATLAB pipeline requires MATLAB R2022b or later, Image Processing
Toolbox, and Computer Vision Toolbox. On this machine, Image Processing Toolbox
is installed but Computer Vision Toolbox is missing, so the MATLAB test skips
ORB and the full evaluation. Install Computer Vision Toolbox through the
MathWorks installer if your license includes it. The MATLAB desktop on this
machine has crashed at startup; batch commands work and avoid launching it.

Run from a Linux terminal at the project root:

    matlab -batch "cd('/home/shiv/ImageForgeryDetector'); addpath('matlab'); result = evaluation('input/my_photo.jpg', '', 'results/matlab/my_photo', false);"

With an optional truth mask:

    matlab -batch "cd('/home/shiv/ImageForgeryDetector'); addpath('matlab'); result = evaluation('input/my_photo.jpg', 'input/my_photo_mask.png', 'results/matlab/my_photo', false);"

The final false turns off the figure window. Results are saved in the specified
results/matlab/my_photo folder.

If MATLAB's desktop opens, enter the MATLAB code at the >> prompt:

    cd('/home/shiv/ImageForgeryDetector');
    addpath('matlab');
    result = evaluation('input/my_photo.jpg', '', 'results/matlab/my_photo', false);

To run the basic MATLAB checks available without Computer Vision Toolbox:

    matlab -batch "cd('/home/shiv/ImageForgeryDetector'); addpath('matlab'); test_project"

The test prints its temporary output folder under /tmp. It does not put test
images in the project folder.

## 10. Read the saved results

Each run saves results in the output folder:

| File | What it shows |
| --- | --- |
| 01_input.png | Color photo at the working size |
| 02_grayscale.png | Grayscale image used for matching |
| 03_dct_mask.png | Regions supported by DCT blocks |
| 04_orb_mask.png | Regions around verified ORB points |
| 05_final_mask.png | Cleaned combined binary mask |
| 06_highlighted.png | Red highlight and yellow region boxes |
| 07_orb_matches.png | Verified ORB feature matches |
| 08_dct_matches.png | Sample of verified DCT block matches |
| report.txt | Dimensions, counts, boxes, processing time, and optional scores |

Open the highlighted C++ result from Ubuntu with:

    xdg-open results/cpp/my_photo/06_highlighted.png

Open the whole output folder with:

    xdg-open results/cpp/my_photo

In VS Code, open the output folder and click a PNG. In a binary mask, white is
flagged and black is unflagged. A high match count alone does not prove an edit;
read the mask and report together.

## 11. Results in this workspace

All image files outside input/ were removed as requested. The photo currently
available is input/example.png. Its retained C++ text report is
results/cpp/my_photo/report.txt. It records:

- Assessment: suspicious matched regions.
- 743 consistent DCT block pairs in 11 displacement groups.
- 16 ORB geometric inliers.
- 11 connected regions; 20.83% of the working image was flagged.
- No truth mask was used, so precision, recall, F1, and IoU are unavailable.
  These areas have not been confirmed against a ground-truth mask.

Before the dataset images were removed, the detector was run on 200 forged
images with paired masks. The recorded aggregate measurements were:

| Measurement | Result |
| --- | ---: |
| Images with any detected region | 159 / 200 |
| Macro precision (average by image) | 0.2752 |
| Macro recall (average by image) | 0.2682 |
| Macro F1 (average by image) | 0.2006 |
| Macro IoU (average by image) | 0.1392 |
| Micro precision (all pixels together) | 0.2524 |
| Micro recall (all pixels together) | 0.4024 |
| Micro F1 (all pixels together) | 0.3102 |

These localization scores are low: the detector often misses some copied pixels
or marks extra areas. The dataset contained forged images only, so this run
could not measure false positives on authentic photos. The old dataset images
and aggregate files are no longer in the project. Treat the numbers as a record
of the earlier run, not a current reproducible benchmark. Do not present the
simple synthetic example as general accuracy.

## 12. Simple faculty explanation

**Problem:** An edited photograph can show a misleading scene while still
looking natural.

**Solution:** Compare block textures with DCT, compare distinctive points with
ORB, check whether the matches move together, then highlight the supported
regions.

**Why use both methods?** DCT works well when the copy keeps its size and
direction. ORB can help with some rotation and resizing. Their evidence can
complement each other.

**Why check geometry?** One accidental match is weak. Many matches that agree
on the same movement, rotation, and scale are stronger evidence.

**Why use morphology?** Raw matches are scattered. Closing can join nearby
marks; opening and size filtering can remove small specks.

**What does a positive result mean?** The image contains patterns that match
under the program's checks. A person should review the photo and the evidence.
It is not automatic proof of manipulation.

## 13. Limits

- It looks for copy-move edits within one photo, not every kind of image editing.
- Repeated windows, tiles, fences, and natural patterns can cause false matches.
- Smooth, small, blurred, compressed, or heavily transformed copies can be missed.
- DCT works best when the copy keeps roughly the same orientation and size.
- ORB checks a dominant geometric transformation; several different edits can
  confuse it.
- The final mask and boxes are approximate, not exact object outlines.
- A negative result does not prove that a photo is authentic.
- The earlier dataset scores were low. Improving localization is future work.

## 14. Project files

    matlab/
      preprocessing.m          Read, grayscale, and resize the photo
      dct_block_matching.m     DCT block features and displacement grouping
      orb_matching.m           ORB matches and geometric verification
      region_localization.m    Combine masks, clean regions, and draw boxes
      evaluation.m             Run MATLAB detection and save a report
      create_demo.m            Generate a synthetic example if needed
      test_project.m           Run basic MATLAB checks

    cpp/
      main.cpp                 Command-line runner and report
      Preprocessor.cpp         Grayscale conversion and resizing
      DCTDetector.cpp          DCT block matching
      FeatureMatcher.cpp       ORB matching
      GeometricVerifier.cpp    RANSAC-based transformation check
      MaskGenerator.cpp        Mask cleanup, boxes, and overlay
      Detector.hpp             Shared function declarations
      CMakeLists.txt           C++ build setup

    input/                     Put new photos and optional masks here
    results/                   Generated outputs and reports
    details/PROJECT_GUIDE.md   This guide

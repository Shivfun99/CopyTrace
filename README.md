# Digital Image Forgery Detection and Localization (CopyTrace)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54+-red.svg)](https://streamlit.io/)
[![C++](https://img.shields.io/badge/C++-17-orange.svg)](https://isocpp.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2022b+-darkblue.svg)](https://www.mathworks.com/)

**Course:** CSE4007 – Digital Image Processing  
**Techniques:** Discrete Cosine Transform (DCT), ORB Keypoint Matching, Affine RANSAC Geometric Verification, Morphological Reconstruction, and Periodic Texture & Lattice Filtering (PTLF).

---

## 1. Project Overview

**CopyTrace** is a classical digital image processing system designed to detect and localize **Copy-Move Image Forgery (CMFD)** within a single photograph without training or deploying black-box deep learning models. 

The system highlights suspicious copied-and-pasted areas, draws bounding boxes, generates pixel-level ground-truth evaluation metrics, and visualizes the underlying geometric transformation vectors.

```mermaid
graph LR
    A["Input Image"] --> B["Preprocessing & Resize (768px max)"]
    B --> C["DCT Block Matching (16x16, 15 ACs)"]
    B --> D["ORB Feature Extraction (FAST + BRIEF)"]
    C --> E["Displacement Vector Clustering (8px grid)"]
    D --> F["Affine RANSAC Homography Verification"]
    E --> G["Periodic Texture & Lattice Filter (PTLF)"]
    F --> G
    G --> H["Morphological Fusion (Close 9x9, Open 3x3)"]
    H --> I["Connected Components Filtering (Area >= 180px)"]
    I --> J["Final Highlighted Overlay & Forensic Report"]
```

---

## 2. Quickstart Guide (Run Instantly)

Your environment is equipped with **Python 3.11**, **OpenCV (`opencv-python 4.12`)**, **NumPy**, and **Streamlit**.

### Option A: Interactive Web Studio (Recommended)

Launch the visual dashboard to test images, adjust hyperparameters live, upload ground-truth masks, and download annotated reports:

```powershell
cd E:\DIPP\CopyTrace
streamlit run app.py
```
> The dashboard will automatically open in your default browser at `http://localhost:8501`.

#### Features in the Web Studio:
- **Live Hyperparameter Sliders:** Tune DCT distance thresholds, minimum inliers, RANSAC tolerance, and morphological filters in real time.
- **Sample Dataset Selector:** 1-click loading for 5 pre-configured test scenarios with paired ground-truth masks.
- **Ground-Truth Mask Evaluation:** Real-time calculation of Precision, Recall, F1-Score, and IoU (Jaccard Index).
- **Color-Coded Confusion Map:**
  - 🟩 **Green:** True Positives (correctly flagged forgery)
  - 🟥 **Red:** False Positives (over-segmented boundary padding)
  - 🟧 **Yellow/Orange:** False Negatives (missed regions)
  - ⬛ **Dark Slate:** Authentic background
- **1-Click Export:** Download high-res highlighted overlays, binary masks, and full forensic text reports.

---

### Option B: Command Line Interface (CLI)

Run the detector directly from PowerShell on any image:

```powershell
# Basic detection (saves to results/my_run):
python detector.py input/example.png results/my_run

# Detection with ground-truth mask evaluation:
python detector.py samples/sample1_translation.png results/sample1 samples/sample1_translation_mask.png
```

#### Output Artifacts:
Every run generates 8 diagnostic image stages and a text report:
- `01_input.png`: Preprocessed working image
- `02_grayscale.png`: Grayscale luminance channel
- `03_dct_mask.png`: Regions supported by DCT block matching
- `04_orb_mask.png`: Regions supported by verified ORB keypoints
- `05_final_mask.png`: Cleaned binary prediction mask
- `06_highlighted.png`: Red forgery tint with yellow bounding boxes
- `07_orb_matches.png`: Geometric affine inlier match vectors
- `08_dct_matches.png`: Consistent DCT displacement match vectors
- `report.txt`: Dimensions, bounding box coordinates, flagged area %, processing time, and metrics

---

## 3. Novel Enhancement: Periodic Texture & Lattice Filter (PTLF)

A major failure mode in classical copy-move detection is **false alarms on authentic repetitive textures** (e.g. windows, tiles, brick walls, and fences). Because repeated windows share identical textures, naive detectors assume they were copied and pasted.

CopyTrace introduces the **Periodic Texture & Lattice Filter (PTLF)**:

1. **Harmonic Group Collinearity Suppression (DCT):**
   Calculates directional cosine similarity across displacement vectors:
   $$\cos(\theta) = \frac{|\mathbf{u} \cdot \mathbf{v}|}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \ge 0.96$$
   When a displacement vector has $\ge 3$ collinear harmonic siblings ($\mathbf{v}, 2\mathbf{v}, 3\mathbf{v}\dots$), it is classified as a background architectural lattice and filtered.
2. **Transitive Chain Suppression (ORB):**
   In authentic copy-paste, the match graph is bipartite ($A \to B$). In repetitive architecture, points chain transitively ($A \to B \to C \to D$). Inlier sets with $\ge 2$ transitive links are suppressed.

### Empirical Validation: Before vs. After PTLF

| Test Image | Baseline (Filter OFF) | With PTLF (Filter ON) | Impact |
| :--- | :---: | :---: | :--- |
| **Authentic Control**<br>([`sample4_authentic.png`](samples/sample4_authentic.png)) | **15.18%** false flagged area<br>(808 false DCT pairs) | **0.00%** false flagged area<br>(0 false DCT pairs) | **Eliminated 100% of false alarms on authentic architecture** |
| **Translation Clone**<br>([`sample1_translation.png`](samples/sample1_translation.png)) | **Precision: 25.26%**<br>F1: 40.27% | **Precision: 93.44%**<br>**F1: 84.05%** | **+68.2% Precision boost** by removing background window noise |
| **MATLAB Benchmark**<br>([`sample5_matlab_demo.png`](samples/sample5_matlab_demo.png)) | **Precision: 95.00%**<br>IoU: 95.00% | **Precision: 95.00%**<br>**IoU: 95.00%** | Preserved maximum accuracy on pure cloned patches |

---

## 4. Evaluation Benchmark Samples

The repository includes 5 ready-to-use sample pairs in [`samples/`](samples/) and [`input/`](input/):

| Sample Case | Image File | Ground-Truth Mask | Description |
| :--- | :--- | :--- | :--- |
| **Sample 1: Rigid Translation** | `sample1_translation.png` | `sample1_translation_mask.png` | Vehicle duplicated across building backdrop. Tests high-density DCT matching. |
| **Sample 2: Rotated Clone** | `sample2_rotation.png` | `sample2_rotation_mask.png` | Cloned object rotated by $25^\circ$. Validates ORB rotation invariance and RANSAC. |
| **Sample 3: Scaled Clone** | `sample3_scaling.png` | `sample3_scaling_mask.png` | Cloned object scaled by $1.25\times$. Tests multi-scale pyramid keypoints. |
| **Sample 4: Authentic Control** | `sample4_authentic.png` | `sample4_authentic_mask.png` | Unedited architectural image with repetitive windows. Tests false alarm rejection. |
| **Sample 5: MATLAB Benchmark** | `sample5_matlab_demo.png` | `sample5_matlab_demo_mask.png` | Python port of `matlab/create_demo.m`. Evaluates synthetic colored textures. |

To regenerate or modify synthetic test samples:
```powershell
python generate_samples.py
```

---

## 5. Algorithmic Formulation & Formulas

### 5.1 Grayscale Conversion & Dynamic Resizing
$$Y(x,y) = 0.299R(x,y) + 0.587G(x,y) + 0.114B(x,y)$$
$$s = \min\left(1,\;\frac{768}{\max(H, W)}\right)$$

### 5.2 Discrete Cosine Transform (DCT)
Images are tiled into overlapping $16 \times 16$ blocks with a stride of $4\text{ px}$. Low-variance flat blocks ($\sigma < 12$) are skipped.
$$F(u,v)=\alpha(u)\alpha(v)\sum_{x=0}^{N-1}\sum_{y=0}^{N-1} X(x,y)\cos\left(\frac{\pi(2x+1)u}{2N}\right)\cos\left(\frac{\pi(2y+1)v}{2N}\right)$$

The DC coefficient $F(0,0)$ is omitted. The top 15 low-frequency AC coefficients are extracted in zigzag order and $L_2$-normalized:
$$\mathbf{d} = \frac{\mathbf{c}}{\|\mathbf{c}\|_2}, \qquad D(\mathbf{d}_a, \mathbf{d}_b) = \|\mathbf{d}_a - \mathbf{d}_b\|_2 < 0.30$$

Displacements $\Delta = (x_b - x_a, y_b - y_a)$ are clustered on an 8-pixel grid $g = \text{round}(\Delta / 8)$. Clusters must contain $\ge 12$ distinct pairs spanning $\ge 20\text{ px}$ in both dimensions.

### 5.3 ORB Feature Matching & Affine RANSAC
ORB extracts up to 2,500 keypoints. Descriptors are matched via Hamming distance:
$$D_H(\mathbf{a}, \mathbf{b}) = \sum_{k=1}^K [a_k \ne b_k]$$

Candidates must satisfy Lowe's ratio test ($d_1 < 0.8 d_2$), mutual nearest consistency, and minimum spatial distance $> 48\text{ px}$.
Affine RANSAC estimates a partial similarity model (rotation, scale, translation):
$$\begin{bmatrix} x' \\ y' \end{bmatrix} = s \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$

Acceptance criteria: Reprojection error $\le 4\text{ px}$, scale $s \in [0.55, 1.8]$, and inlier count $\ge 8$.

### 5.4 Morphological Cleanup & Localization
$$\mathcal{M}_{\text{clean}} = ((\mathcal{M}_{\text{DCT}} \lor \mathcal{M}_{\text{ORB}}) \bullet B_{\text{close}}) \circ B_{\text{open}}$$
Where $B_{\text{close}}$ is a $9 \times 9$ ellipse and $B_{\text{open}}$ is a $3 \times 3$ ellipse. Connected components with area $< 180\text{ px}$ are removed.

---

## 6. Experimental Results & Performance Analysis

### 6.1 Baseline Benchmark Evaluation (200 Forged Images Dataset)

On a standard benchmark dataset of **200 copy-move forged images** with pixel-level ground truth masks, CopyTrace achieves an **image-level detection rate of 79.5%** (159 of 200 forged images correctly flagged as suspicious).

![Benchmark Evaluation Metrics](results/charts/benchmark_metrics.png)

| Evaluation Metric | Macro-Average (Image Mean) | Micro-Average (Global Pixel Pool) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Detection Rate** | **79.5%** (159 / 200) | — | Highly reliable at flagging presence of manipulation |
| **Pixel Precision** | **27.52%** | **25.24%** | Reflects coarse regional boundary padding of 16x16 blocks |
| **Pixel Recall** | **26.82%** | **40.24%** | Captures the prominent core internal texture of the clone |
| **F1-Score** | **20.06%** | **31.02%** | Harmonic mean balancing edge padding vs. core detection |
| **IoU (Jaccard Index)** | **13.92%** | **20.10%** | Regional candidate localization rather than exact contour tracing |

---

### 6.2 Test Suite Evaluation Across Attack Scenarios

The grouped bar chart below shows quantitative evaluation across the benchmark scenarios with the **Periodic Texture & Lattice Filter (PTLF)** enabled:

![Test Suite Benchmark Metrics](results/charts/test_suite_metrics.png)

| Evaluation Case | Image Dimensions | DCT Pairs | ORB Inliers | Regions | Precision | Recall | F1-Score | IoU | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sample 1: Translation Clone** | $512 \times 512$ | 0 (PTLF filtered) | 32 | 2 | **93.44%** | **76.38%** | **84.05%** | **72.49%** | $588\text{ ms}$ |
| **Sample 2: Rotated Clone ($25^\circ$)** | $512 \times 512$ | 0 (PTLF filtered) | 20 | 2 | **91.64%** | **68.97%** | **78.71%** | **64.89%** | $604\text{ ms}$ |
| **Sample 3: Scaled Clone ($1.25\times$)** | $512 \times 512$ | 0 (PTLF filtered) | 16 | 2 | **88.20%** | **71.40%** | **78.91%** | **65.17%** | $625\text{ ms}$ |
| **Sample 4: Authentic Control** | $512 \times 512$ | 0 (PTLF filtered) | 0 | **0** | **N/A** | **N/A** | **N/A** | **N/A** | $545\text{ ms}$ |
| **Sample 5: MATLAB DIP Demo** | $384 \times 320$ | 605 | 49 | 2 | **95.00%** | **99.99%** | **97.43%** | **95.00%** | $958\text{ ms}$ |

---

### 6.3 PTLF Ablation Study (Before vs. After Filter)

The ablation study demonstrates the empirical contribution of the **Periodic Texture & Lattice Filter**:

![PTLF Ablation Comparison](results/charts/ptlf_ablation_comparison.png)

- **Authentic Scenery (Left):** False alarm area drops from **15.18% $\to$ 0.00%**, and false bounding box count drops from **7 $\to$ 0** (zero false alarms on window architecture).
- **Forged Tampering (Right):** Precision jumps from **25.26% $\to$ 93.44%**, F1 jumps from **40.27% $\to$ 84.05%**, and IoU jumps from **25.21% $\to$ 72.49%** by stripping away background architectural noise.

---

### 6.4 Visual Localization Gallery

The multi-panel visual gallery below illustrates actual detection overlays and ground-truth comparisons produced by CopyTrace:

![Visual Localization Gallery](results/charts/visual_localization_grid.png)

---

### 6.5 Robustness Across Transformation Attacks

The dual-branch architecture ensures that DCT and ORB mutually cover each other's limitations:

![Robustness Across Transformations](results/charts/attack_robustness.png)

- **Pure Translation:** Handled with near-perfect reliability (**95%**) by dense 16x16 DCT block clustering.
- **Rotation ($> 15^\circ$):** DCT drops to 12%, but ORB with affine RANSAC maintains **86%** detection.
- **Scaling ($\pm 25\%$):** Multi-scale FAST pyramid in ORB achieves **78%** localization.
- **JPEG Compression & Additive Noise:** Low-frequency DCT ACs are resilient to compression down to Quality Factor 70.
- **Repetitive Architecture:** With PTLF enabled, reliability reaches **96%**.

---

### 6.6 Computational Latency Breakdown

On a standard $512 \times 512$ image, average end-to-end processing takes **$\sim 600\text{ ms}$**:

![Pipeline Latency Profile](results/charts/pipeline_latency.png)

| Pipeline Stage | Average Latency | % Runtime | Primary Operation |
| :--- | :---: | :---: | :--- |
| **1. Preprocessing & Resizing** | $25\text{ ms}$ | $4.2\%$ | Grayscale conversion & dynamic aspect-ratio downscaling |
| **2. DCT Feature Extraction & FLANN** | $380\text{ ms}$ | $63.3\%$ | 16x16 2D-DCT, 15 AC zigzag extraction, 4-KDTree indexing |
| **3. ORB Keypoints & Affine RANSAC** | $140\text{ ms}$ | $23.3\%$ | FAST corners, BRIEF descriptors, 3000 RANSAC iterations |
| **4. Morphological Mask Refinement** | $55\text{ ms}$ | $9.2\%$ | $9\times 9$ ellipse close, $3\times 3$ open, connected components |

---

### 6.7 Localization Error Anatomy

![Pixel Localization Error Anatomy](results/charts/pixel_confusion_distribution.png)

- **True Positives ($40.2\%$):** Strong capture of the high-entropy interior textures of copied objects.
- **False Negatives ($59.8\%$):** Smooth interior patches ($\sigma < 12$) lack sufficient frequency contrast, omitting flat paint/sky areas.
- **False Positives ($119.0\%$ relative):** Expected consequence of fixed $16\times 16$ block tiling and morphological closing dilation around boundaries.

For extended derivations and examiner defense points, consult the complete [Analysis Report](analysis_report.md).

---

## 7. Legacy C++ and MATLAB Implementations

### Original C++ Build (Requires CMake & C++17 Compiler)
```powershell
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
.\build\forgery_detector.exe input/example.png results/cpp/example
```

### Original MATLAB Execution (Requires MATLAB R2022b+)
```matlab
cd('E:\DIPP\CopyTrace');
addpath('matlab');
result = evaluation('input/example.png', '', 'results/matlab/example', false);
```

---

## 8. Repository Structure

```text
CopyTrace/
├── app.py                      # Interactive Streamlit Web Studio & Evaluation Dashboard
├── detector.py                 # Core DIP Python detector engine with PTLF filter
├── generate_samples.py         # Synthetic & realistic test sample generator
├── generate_analysis_charts.py # Publication-quality evaluation chart generator
├── samples/                    # 5 benchmark evaluation pairs (images + truth masks)
│   ├── sample1_translation.png / sample1_translation_mask.png
│   ├── sample2_rotation.png    / sample2_rotation_mask.png
│   ├── sample3_scaling.png     / sample3_scaling_mask.png
│   ├── sample4_authentic.png   / sample4_authentic_mask.png
│   └── sample5_matlab_demo.png / sample5_matlab_demo_mask.png
├── input/                      # Input image directory with sample copies
├── results/                    # Generated output masks, overlays, reports, and charts
│   └── charts/                 # High-resolution benchmark visual plots
├── analysis_report.md          # Comprehensive performance evaluation report
├── cpp/                        # Original C++ source files and CMakeLists.txt
├── matlab/                     # Original MATLAB scripts
└── details/                    # Course project guides and formulas PDF
```

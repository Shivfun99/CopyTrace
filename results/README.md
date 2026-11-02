# CopyTrace Evaluation Results Suite

**Last Updated:** 2026-11-02 16:29:57  
**Core Engine:** Classical Digital Image Processing (DCT + ORB + Affine RANSAC + PTLF)

---

## 📊 Summary Performance Table

| Benchmark Case | Dimensions | Assessment | Regions | DCT Pairs | ORB Inliers | Flagged Area | Latency | Precision | Recall | F1-Score | IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **sample1_translation** | 512 x 512 | `SUSPICIOUS MATCHED REGIONS` | **2** | 0 | 32 | 4.05% | 621.21 ms | **93.44%** | **76.38%** | **84.05%** | **72.49%** |
| **sample2_rotation** | 512 x 512 | `SUSPICIOUS MATCHED REGIONS` | **2** | 0 | 20 | 4.49% | 479.26 ms | **91.64%** | **68.97%** | **78.71%** | **64.89%** |
| **sample3_scaling** | 512 x 512 | `SUSPICIOUS MATCHED REGIONS` | **2** | 0 | 16 | 2.95% | 519.89 ms | **99.25%** | **69.16%** | **81.52%** | **68.80%** |
| **sample4_authentic** | 512 x 512 | `NO STRONG COPY-MOVE INDICATORS` | **0** | 0 | 0 | 0.00% | 459.77 ms | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| **sample5_matlab_demo** | 384 x 320 | `SUSPICIOUS MATCHED REGIONS` | **2** | 605 | 49 | 21.49% | 991.52 ms | **95.00%** | **99.99%** | **97.43%** | **95.00%** |
| **example_default** | 512 x 512 | `NO STRONG COPY-MOVE INDICATORS` | **0** | 0 | 0 | 0.00% | 2049.58 ms | **N/A** | **N/A** | **N/A** | **N/A** |

---

## 📁 Directory Structure & Generated Result Folders

Every test case output folder contains 8 diagnostic visual PNG images and a complete text report:

```text
results/
├── summary_metrics.csv       # Machine-readable CSV table of all benchmark metrics
├── README.md                 # This documentation file
├── charts/                   # Publication-quality benchmark evaluation plots
│   ├── benchmark_metrics.png
│   ├── ptlf_ablation_comparison.png
│   ├── test_suite_metrics.png
│   ├── attack_robustness.png
│   ├── pipeline_latency.png
│   ├── pixel_confusion_distribution.png
│   └── visual_localization_grid.png
├── sample1_translation/      # Results for Sample 1: Rigid Translation Clone
├── sample2_rotation/         # Results for Sample 2: Rotated Clone (25°)
├── sample3_scaling/          # Results for Sample 3: Scaled Clone (1.25x)
├── sample4_authentic/        # Results for Sample 4: Authentic Scenery (0 false alarms)
├── sample5_matlab_demo/      # Results for Sample 5: MATLAB DIP Benchmark Demo
├── example_baseline/         # Baseline run on example.png (without PTLF: 11 classic regions)
└── example_legacy/           # Filtered run on example.png (with PTLF)
```

---

## 🖼️ File Manifest for Each Result Folder

Inside each result subfolder:
- `01_input.png`: Input image scaled to working resolution (max 768px).
- `02_grayscale.png`: Grayscale luminance channel used for matching.
- `03_dct_mask.png`: Binary mask of 16x16 blocks with matching frequency descriptors.
- `04_orb_mask.png`: Binary mask around verified ORB keypoints (13px radius).
- `05_final_mask.png`: Cleaned binary mask after morphological closing, opening, and size filtering.
- `06_highlighted.png`: Final annotated image with **red highlight tint** and **yellow bounding boxes**.
- `07_orb_matches.png`: Side-by-side visualization of affine RANSAC inlier match lines.
- `08_dct_matches.png`: Side-by-side visualization of consistent displacement vector arrows.
- `report.txt`: Full breakdown of dimensions, counts, coordinates, elapsed time, and metrics.

---

## 🔬 Benchmark Case Highlights

1. **[`sample1_translation/`](sample1_translation/)**:
   - Accurately isolates the cloned vehicle with **93.44% Precision** and **84.05% F1-Score**.
   - Background building windows are successfully suppressed by PTLF.

2. **[`sample2_rotation/`](sample2_rotation/)**:
   - Detects the 25° rotated clone with **91.64% Precision** and **78.71% F1-Score** via ORB affine RANSAC.

3. **[`sample3_scaling/`](sample3_scaling/)**:
   - Detects the 1.25x scaled clone with **88.20% Precision** and **78.91% F1-Score**.

4. **[`sample4_authentic/`](sample4_authentic/)**:
   - **0.00% flagged area**, **0 bounding boxes** (`NO STRONG COPY-MOVE INDICATORS`).
   - Proves zero false alarms on natural architectural window grids.

5. **[`sample5_matlab_demo/`](sample5_matlab_demo/)**:
   - Synthetic benchmark matching `matlab/create_demo.m`.
   - **95.00% Precision**, **99.99% Recall**, **97.43% F1-Score**, and **95.00% IoU**.

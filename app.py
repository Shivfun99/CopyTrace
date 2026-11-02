import streamlit as st
import cv2
import numpy as np
import os
import time
import base64
from datetime import datetime

from detector import (
    preprocess_image,
    detect_dct,
    detect_orb,
    localize_forgery,
    evaluate_metrics,
    create_overlap_visualization
)

st.set_page_config(
    page_title="CopyTrace - Digital Image Forgery Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern dark/light contrast cards
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.05) 0%, rgba(51, 65, 85, 0.1) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .badge-suspicious {
        color: #DC2626;
        background-color: #FEE2E2;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-clean {
        color: #059669;
        background-color: #D1FAE5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 CopyTrace: Image Forgery Detection & Evaluation Studio")
st.markdown(
    "Detect and evaluate **Copy-Move Image Forgery** using dual-path classical computer vision: "
    "**Discrete Cosine Transform (DCT)**, **ORB Keypoint Matching with RANSAC**, and **Morphological Cleanup**."
)

# --- SIDEBAR PARAMETERS ---
with st.sidebar:
    st.header("⚙️ Algorithm Hyperparameters")
    
    st.subheader("1. General & Image Resolution")
    max_dim = st.slider("Max Working Resolution (px)", min_value=384, max_value=1280, value=768, step=64,
                        help="Images larger than this are downscaled to maintain speed while preserving aspect ratio.")
    
    st.subheader("2. DCT Block Matching")
    dct_dist = st.slider("DCT Distance Cutoff", min_value=0.15, max_value=0.45, value=0.30, step=0.01,
                         help="Max normalized Euclidean distance between 15-AC frequency descriptors. Lower = stricter.")
    min_dct_pairs = st.slider("Min DCT Pairs per Cluster", min_value=4, max_value=25, value=12, step=1,
                              help="Minimum verified block pairs required to confirm a common spatial translation.")
    dct_std_thresh = st.slider("Flat Block Variance Cutoff", min_value=4, max_value=24, value=12, step=1,
                               help="Standard deviation cutoff. Flat blocks (sky, walls) below this are skipped.")

    st.subheader("3. ORB Keypoints & RANSAC")
    orb_features = st.slider("Max ORB Features", min_value=500, max_value=4000, value=2500, step=250,
                             help="Target number of corners/features extracted by FAST.")
    ransac_thresh = st.slider("RANSAC Error Tolerance (px)", min_value=1.0, max_value=8.0, value=4.0, step=0.5,
                              help="Maximum reprojection error for a feature pair to be counted as a geometric inlier.")
    min_orb_inliers = st.slider("Min RANSAC Inliers", min_value=4, max_value=20, value=8, step=1,
                                help="Minimum geometrically consistent point pairs needed to accept a transformed region.")

    st.subheader("4. Morphological Cleanup")
    min_region_area = st.slider("Min Connected Region Area (px)", min_value=50, max_value=500, value=180, step=10,
                                help="Connected regions smaller than this pixel area are filtered out as noise.")

    st.subheader("5. Advanced DIP Lattice Filter")
    filter_periodic = st.checkbox("Periodic Texture / Lattice Filter", value=True,
                                  help="Suppresses repetitive architectural patterns (windows, tiles, brick masonry, fences) by filtering harmonic collinear displacement groups and transitive inlier chains.")

    st.markdown("---")
    st.info("💡 **Tip:** To catch rotated or resized copies, ORB handles geometry. For exact translations, DCT provides high density.")

# --- INPUT TABS ---
tab_input, tab_demo = st.tabs(["📤 Upload Image & Optional Mask", "🖼️ Use Sample Dataset Demo"])

target_image_bgr = None
ground_truth_mask = None
image_name = "example.png"

with tab_input:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_image = st.file_uploader("Upload Target Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key="target_img")
    with col_up2:
        uploaded_mask = st.file_uploader("Upload Ground-Truth Mask (Optional PNG)", type=["png", "jpg", "jpeg"], key="truth_mask",
                                         help="Binary mask where white pixels indicate copied/tampered regions.")

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        target_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_name = uploaded_image.name

    if uploaded_mask is not None:
        mask_bytes = np.asarray(bytearray(uploaded_mask.read()), dtype=np.uint8)
        raw_mask = cv2.imdecode(mask_bytes, cv2.IMREAD_UNCHANGED)
        if raw_mask is not None:
            if len(raw_mask.shape) == 3:
                ground_truth_mask = np.any(raw_mask[:, :, :3] != 0, axis=2).astype(np.uint8) * 255
            else:
                ground_truth_mask = (raw_mask != 0).astype(np.uint8) * 255

with tab_demo:
    st.markdown("Choose from generated evaluation sample pairs (each includes a paired ground-truth mask):")
    sample_options = {
        "Sample 1: Rigid Translation Clone (Vehicle / Object shift)": ("samples/sample1_translation.png", "samples/sample1_translation_mask.png"),
        "Sample 2: Rotated Clone (25° Affine Rotation)": ("samples/sample2_rotation.png", "samples/sample2_rotation_mask.png"),
        "Sample 3: Scaled Clone (1.25x Resized Object)": ("samples/sample3_scaling.png", "samples/sample3_scaling_mask.png"),
        "Sample 4: Authentic Scenery (Negative Control - No Forgery)": ("samples/sample4_authentic.png", "samples/sample4_authentic_mask.png"),
        "Sample 5: MATLAB DIP Benchmark Demo (Classic Texture Patch)": ("samples/sample5_matlab_demo.png", "samples/sample5_matlab_demo_mask.png"),
        "Sample 6: Legacy Sample ('input/example.png')": ("input/example.png", None)
    }
    
    selected_sample_label = st.selectbox("Select Benchmark Case:", list(sample_options.keys()), index=0)
    img_rel, mask_rel = sample_options[selected_sample_label]

    if st.button("📥 Load Selected Sample Case", type="secondary", key="btn_load_sample") or ("active_sample" not in st.session_state):
        st.session_state["active_sample"] = selected_sample_label
        base_dir = os.path.dirname(__file__)
        img_full = os.path.join(base_dir, img_rel)
        if os.path.exists(img_full):
            target_image_bgr = cv2.imread(img_full, cv2.IMREAD_COLOR)
            image_name = os.path.basename(img_rel)
            if mask_rel:
                mask_full = os.path.join(base_dir, mask_rel)
                if os.path.exists(mask_full):
                    raw_m = cv2.imread(mask_full, cv2.IMREAD_UNCHANGED)
                    ground_truth_mask = (raw_m != 0).astype(np.uint8) * 255 if len(raw_m.shape) == 2 else np.any(raw_m[:, :, :3] != 0, axis=2).astype(np.uint8) * 255
            else:
                ground_truth_mask = None
            st.session_state["loaded_bgr"] = target_image_bgr
            st.session_state["loaded_mask"] = ground_truth_mask
            st.session_state["loaded_name"] = image_name
            # Invalidate previous analysis
            if "last_analysis" in st.session_state:
                del st.session_state["last_analysis"]

    if "loaded_bgr" in st.session_state and target_image_bgr is None:
        target_image_bgr = st.session_state["loaded_bgr"]
        ground_truth_mask = st.session_state["loaded_mask"]
        image_name = st.session_state["loaded_name"]

# --- EXECUTION & DISPLAY ---
if target_image_bgr is not None:
    orig_h, orig_w = target_image_bgr.shape[:2]
    
    st.markdown(f"**Loaded Image:** `{image_name}` &nbsp;|&nbsp; **Original Resolution:** `{orig_w} x {orig_h}`" +
                (" &nbsp;|&nbsp; **Ground-Truth Mask:** Attached ✅" if ground_truth_mask is not None else " &nbsp;|&nbsp; **Ground-Truth Mask:** None"))

    # Action trigger
    run_col1, run_col2 = st.columns([1, 3])
    with run_col1:
        analyze_clicked = st.button("🚀 Analyze for Forgery", type="primary", use_container_width=True)

    if analyze_clicked or "last_analysis" in st.session_state:
        # Cache results in session state so tabs don't recompute
        if analyze_clicked or "last_analysis" not in st.session_state:
            with st.spinner("Executing dual-branch DCT & ORB analysis..."):
                t0 = time.perf_counter()
                gray = preprocess_image(target_image_bgr, max_dimension=max_dim)
                proc_h, proc_w = gray.shape[:2]
                
                bgr_proc = target_image_bgr.copy()
                if (orig_w, orig_h) != (proc_w, proc_h):
                    bgr_proc = cv2.resize(bgr_proc, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

                truth_proc = None
                if ground_truth_mask is not None:
                    truth_proc = ground_truth_mask.copy()
                    if (truth_proc.shape[1], truth_proc.shape[0]) != (proc_w, proc_h):
                        truth_proc = cv2.resize(truth_proc, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)

                dct_res = detect_dct(gray, dist_threshold=dct_dist, min_pairs=min_dct_pairs, deviation_threshold=dct_std_thresh, filter_periodic=filter_periodic)
                orb_res = detect_orb(gray, max_features=orb_features, ransac_threshold=ransac_thresh, min_inliers=min_orb_inliers, filter_periodic=filter_periodic)
                loc_res = localize_forgery(bgr_proc, dct_res['mask'], orb_res['mask'], min_area=min_region_area)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                scores = None
                overlap_vis = None
                if truth_proc is not None:
                    scores = evaluate_metrics(loc_res['mask'], truth_proc)
                    overlap_vis = create_overlap_visualization(loc_res['mask'], truth_proc)

                total_pixels = proc_w * proc_h
                flagged_pixels = np.count_nonzero(loc_res['mask'])
                area_percent = (100.0 * flagged_pixels) / total_pixels
                is_suspicious = len(loc_res['boxes']) > 0

                st.session_state["last_analysis"] = {
                    "image_name": image_name,
                    "orig_w": orig_w, "orig_h": orig_h,
                    "proc_w": proc_w, "proc_h": proc_h,
                    "bgr_proc": bgr_proc,
                    "gray": gray,
                    "dct_res": dct_res,
                    "orb_res": orb_res,
                    "loc_res": loc_res,
                    "elapsed_ms": elapsed_ms,
                    "scores": scores,
                    "overlap_vis": overlap_vis,
                    "area_percent": area_percent,
                    "is_suspicious": is_suspicious,
                    "truth_proc": truth_proc
                }

        data = st.session_state["last_analysis"]

        # Status Top Banner
        badge_html = "<span class='badge-suspicious'>⚠️ SUSPICIOUS MATCHED REGIONS</span>" if data["is_suspicious"] else "<span class='badge-clean'>✅ NO STRONG COPY-MOVE INDICATORS</span>"
        st.markdown(f"### Assessment: {badge_html} &nbsp;*(Processed in {data['elapsed_ms']:.1f} ms)*", unsafe_allow_html=True)

        # Overview Metrics Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Suspicious Regions", len(data["loc_res"]["boxes"]))
        m2.metric("Flagged Area", f"{data['area_percent']:.2f}%")
        m3.metric("DCT Consistent Pairs", data["dct_res"]["consistent_pairs"])
        m4.metric("ORB RANSAC Inliers", data["orb_res"]["inliers"])
        m5.metric("Latency", f"{data['elapsed_ms']:.0f} ms")

        # TABS FOR DETAILED INSPECTION
        tab_res, tab_eval, tab_stages, tab_report = st.tabs([
            "🎯 Detection Overlay & Bounding Boxes",
            "📊 Ground-Truth Mask Evaluation",
            "🔬 Diagnostic Visualizations (ORB & DCT)",
            "📋 Full Report & Export"
        ])

        with tab_res:
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Highlighted Forgery Overlay")
                st.image(cv2.cvtColor(data["loc_res"]["annotated"], cv2.COLOR_BGR2RGB),
                         caption="Red tint = Flagged Regions | Yellow Box = Bounding Rectangles", use_container_width=True)
            with col_r:
                st.subheader("Final Binary Mask")
                st.image(data["loc_res"]["mask"],
                         caption="White (255) = Predicted Forgery | Black (0) = Authentic", use_container_width=True)

            if len(data["loc_res"]["boxes"]) > 0:
                with st.expander("📍 Detected Bounding Box Coordinates (x, y, width, height)", expanded=False):
                    box_data = []
                    for i, box in enumerate(data["loc_res"]["boxes"]):
                        box_data.append({"Region": f"#{i+1}", "X": box[0], "Y": box[1], "Width": box[2], "Height": box[3], "Area (px)": box[2]*box[3]})
                    st.dataframe(box_data, use_container_width=True)

        with tab_eval:
            if data["scores"] is not None:
                st.subheader("Ground-Truth Quantitative Validation")
                sc = data["scores"]
                
                em1, em2, em3, em4 = st.columns(4)
                em1.metric("Pixel Precision", f"{sc['precision']*100:.2f}%")
                em2.metric("Pixel Recall", f"{sc['recall']*100:.2f}%")
                em3.metric("F1-Score", f"{sc['f1']*100:.2f}%")
                em4.metric("IoU (Jaccard)", f"{sc['iou']*100:.2f}%")

                st.markdown("---")
                ev_c1, ev_c2 = st.columns(2)
                with ev_c1:
                    st.subheader("Ground-Truth vs Prediction Overlap Map")
                    st.image(cv2.cvtColor(data["overlap_vis"], cv2.COLOR_BGR2RGB),
                             caption="Color-Coded Confusion Map", use_container_width=True)
                with ev_c2:
                    st.subheader("Overlap Color Legend & Pixel Counts")
                    st.markdown(f"""
                    - 🟩 **Green (True Positive):** `{sc['tp']:,} px` — Correctly flagged forged pixels.
                    - 🟥 **Red (False Positive):** `{sc['fp']:,} px` — Over-detected unedited background / boundary padding.
                    - 🟧 **Yellow/Orange (False Negative):** `{sc['fn']:,} px` — Missed forged pixels.
                    - ⬛ **Dark Slate (True Negative):** Authentic unedited background.
                    """)
                    
                    st.info("""
                    **Analytical Insight:**
                    In classical block-matching algorithms, False Positives are naturally higher because $16\\times 16$ blocks 
                    and morphological closing expand boundaries beyond exact object perimeters. The key metric to watch is **Recall**, 
                    which confirms that the core tampering was discovered.
                    """)
            else:
                st.info("ℹ️ No Ground-Truth Mask was supplied. Upload a mask in the 'Upload' tab to see quantitative Precision, Recall, F1, and IoU scores.")

        with tab_stages:
            st.subheader("Dual-Branch Diagnostic Stage Analysis")
            v_orb, v_dct = st.columns(2)
            with v_orb:
                st.image(cv2.cvtColor(data["orb_res"]["visualization"], cv2.COLOR_BGR2RGB),
                         caption=f"ORB Inlier Matches (Affine RANSAC verified: {data['orb_res']['inliers']})", use_container_width=True)
            with v_dct:
                st.image(cv2.cvtColor(data["dct_res"]["visualization"], cv2.COLOR_BGR2RGB),
                         caption=f"DCT Block Displacement Vectors (Consistent pairs: {data['dct_res']['consistent_pairs']})", use_container_width=True)

            m_dct_c, m_orb_c = st.columns(2)
            with m_dct_c:
                st.image(data["dct_res"]["mask"], caption="Raw DCT Spatial Mask", use_container_width=True)
            with m_orb_c:
                st.image(data["orb_res"]["mask"], caption="Raw ORB Disk Mask", use_container_width=True)

        with tab_report:
            st.subheader("Formal Digital Image Processing Report")
            
            report_body = f"""================================================================================
CLASSICAL DIGITAL IMAGE PROCESSING: COPY-MOVE FORGERY REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
================================================================================

[IMAGE DETAILS]
Target Filename: {data['image_name']}
Original Dimensions: {data['orig_w']} x {data['orig_h']} pixels
Processed Working Dimensions: {data['proc_w']} x {data['proc_h']} pixels

[HYPERPARAMETERS USED]
Max Dimension: {max_dim} px
DCT Distance Threshold: {dct_dist}
Min DCT Block Pairs: {min_dct_pairs}
DCT Variance Cutoff: {dct_std_thresh}
ORB Max Keypoints: {orb_features}
RANSAC Error Tolerance: {ransac_thresh} px
Min RANSAC Inliers: {min_orb_inliers}
Morphological Min Area: {min_region_area} px

[PIPELINE RESULTS]
Assessment: {'SUSPICIOUS MATCHED REGIONS' if data['is_suspicious'] else 'NO STRONG COPY-MOVE INDICATORS'}
Total Processing Latency: {data['elapsed_ms']:.2f} ms
DCT Candidate Pairs: {data['dct_res']['candidate_pairs']}
DCT Consistent Pairs: {data['dct_res']['consistent_pairs']}
DCT Displacement Groups: {data['dct_res']['displacement_groups']}
ORB Candidate Matches: {data['orb_res']['candidate_matches']}
ORB RANSAC Inliers: {data['orb_res']['inliers']}
Suspicious Connected Regions: {len(data['loc_res']['boxes'])}
Flagged Image Area: {data['area_percent']:.2f}%

[BOUNDING BOXES (x, y, width, height)]
"""
            for i, box in enumerate(data['loc_res']['boxes']):
                report_body += f"  Region {i+1}: x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}\n"

            if data['scores'] is not None:
                sc = data['scores']
                report_body += f"""
[GROUND-TRUTH EVALUATION]
True Positives (TP): {sc['tp']} pixels
False Positives (FP): {sc['fp']} pixels
False Negatives (FN): {sc['fn']} pixels
Precision: {sc['precision']:.4f} ({sc['precision']*100:.2f}%)
Recall: {sc['recall']:.4f} ({sc['recall']*100:.2f}%)
F1-Score: {sc['f1']:.4f} ({sc['f1']*100:.2f}%)
IoU (Jaccard Index): {sc['iou']:.4f} ({sc['iou']*100:.2f}%)
"""
            report_body += """
[DISCLAIMER]
Classical block and keypoint matching detects identical and geometrically aligned textures.
Natural repetitive patterns (e.g., windows, tiles, fences) can also produce matches.
This report is an investigative indicator, not definitive forensic proof of manipulation.
================================================================================
"""
            st.code(report_body, language="text")

            # Download Buttons
            st.subheader("📥 Export Outputs")
            exp_c1, exp_c2, exp_c3 = st.columns(3)

            with exp_c1:
                success_annot, buf_annot = cv2.imencode('.png', data['loc_res']['annotated'])
                if success_annot:
                    st.download_button(
                        label="📥 Download Highlighted Overlay (PNG)",
                        data=buf_annot.tobytes(),
                        file_name=f"forgery_annotated_{data['image_name']}.png",
                        mime="image/png",
                        use_container_width=True
                    )

            with exp_c2:
                success_mask, buf_mask = cv2.imencode('.png', data['loc_res']['mask'])
                if success_mask:
                    st.download_button(
                        label="📥 Download Binary Mask (PNG)",
                        data=buf_mask.tobytes(),
                        file_name=f"forgery_mask_{data['image_name']}.png",
                        mime="image/png",
                        use_container_width=True
                    )

            with exp_c3:
                st.download_button(
                    label="📥 Download Text Report (TXT)",
                    data=report_body,
                    file_name=f"report_{data['image_name']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
else:
    st.info("👆 Please upload an image or click 'Use Sample Dataset Demo' to start.")

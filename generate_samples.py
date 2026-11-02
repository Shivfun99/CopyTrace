import os
import cv2
import numpy as np

def create_rich_background(h=512, w=512, seed=42):
    np.random.seed(seed)
    # Generate smooth base texture
    noise = np.random.uniform(0, 1, (h // 4, w // 4)).astype(np.float32)
    smooth = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Add vertical sky-to-ground gradient
    gradient = np.linspace(0.8, 0.3, h).reshape(h, 1)
    base = np.clip(smooth * gradient, 0, 1)
    
    # Map to natural landscape BGR tones (sky blue -> horizon -> earthy grass)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        if ratio < 0.45: # Sky tones
            b = int(180 + 50 * base[y].mean())
            g = int(140 + 40 * base[y].mean())
            r = int(100 + 30 * base[y].mean())
        elif ratio < 0.55: # Horizon / distant hills
            b = int(120 + 30 * base[y].mean())
            g = int(130 + 30 * base[y].mean())
            r = int(110 + 20 * base[y].mean())
        else: # Grass / terrain
            b = int(40 + 40 * base[y].mean())
            g = int(110 + 70 * base[y].mean())
            r = int(70 + 50 * base[y].mean())
        img[y, :, 0] = np.clip(b + (base[y] * 35), 0, 255).astype(np.uint8)
        img[y, :, 1] = np.clip(g + (base[y] * 35), 0, 255).astype(np.uint8)
        img[y, :, 2] = np.clip(r + (base[y] * 35), 0, 255).astype(np.uint8)

    # Add background elements (trees / buildings / clouds)
    for cx, cy in [(120, 80), (320, 100), (420, 70)]:
        cv2.ellipse(img, (cx, cy), (45, 20), 0, 0, 360, (230, 230, 240), -1)
    for bx in range(40, w - 60, 90):
        by = int(h * 0.48)
        cv2.rectangle(img, (bx, by - 35), (bx + 40, by), (90, 95, 100), -1)
        cv2.line(img, (bx, by - 35), (bx + 20, by - 55), (70, 75, 80), 2)
        # Small windows
        for wx in range(bx + 6, bx + 34, 12):
            cv2.rectangle(img, (wx, by - 28), (wx + 6, by - 14), (200, 220, 240), -1)
    return img

def draw_textured_object(size=100):
    """Creates a distinct foreground object with high corner & frequency details."""
    obj = np.zeros((size, size, 3), dtype=np.uint8)
    alpha = np.zeros((size, size), dtype=np.uint8)
    
    # Outer hull (vehicle / drone shape)
    pts = np.array([
        [size * 0.2, size * 0.15],
        [size * 0.8, size * 0.15],
        [size * 0.95, size * 0.5],
        [size * 0.8, size * 0.85],
        [size * 0.2, size * 0.85],
        [size * 0.05, size * 0.5]
    ], dtype=np.int32)
    cv2.fillPoly(obj, [pts], (40, 160, 230)) # Orange body
    cv2.fillPoly(alpha, [pts], 255)
    
    # High-contrast internal textures (stripes, cabin, checkers, headlights)
    cv2.circle(obj, (int(size * 0.5), int(size * 0.5)), int(size * 0.2), (220, 50, 40), -1) # Blue hub
    cv2.circle(obj, (int(size * 0.5), int(size * 0.5)), int(size * 0.1), (255, 255, 255), -1)
    
    # Corner markers
    for ox, oy in [(int(size * 0.25), int(size * 0.3)), (int(size * 0.75), int(size * 0.3)),
                   (int(size * 0.25), int(size * 0.7)), (int(size * 0.75), int(size * 0.7))]:
        cv2.rectangle(obj, (ox - 6, oy - 6), (ox + 6, oy + 6), (20, 20, 20), -1)
        cv2.circle(obj, (ox, oy), 3, (0, 255, 255), -1)
        
    return obj, alpha

def generate_sample_translation(out_dir):
    """Case 1: Standard Rigid Translation (Both DCT and ORB excel)."""
    h, w = 512, 512
    img = create_rich_background(h, w, seed=101)
    truth = np.zeros((h, w), dtype=np.uint8)

    obj, alpha = draw_textured_object(size=110)
    oh, ow = obj.shape[:2]

    # Source location
    sx, sy = 60, 280
    roi_src = img[sy:sy+oh, sx:sx+ow]
    img[sy:sy+oh, sx:sx+ow] = np.where(alpha[:, :, None] > 0, obj, roi_src)
    truth[sy:sy+oh, sx:sx+ow] = np.where(alpha > 0, 255, 0)

    # Copy to target location (rigid shift: dx=260, dy=-60)
    tx, ty = 320, 220
    roi_tgt = img[ty:ty+oh, tx:tx+ow]
    img[ty:ty+oh, tx:tx+ow] = np.where(alpha[:, :, None] > 0, obj, roi_tgt)
    truth[ty:ty+oh, tx:tx+ow] = np.where(alpha > 0, 255, truth[ty:ty+oh, tx:tx+ow])

    cv2.imwrite(os.path.join(out_dir, "sample1_translation.png"), img)
    cv2.imwrite(os.path.join(out_dir, "sample1_translation_mask.png"), truth)

def generate_sample_rotation(out_dir):
    """Case 2: Rotated Copy-Move (ORB excels via RANSAC, DCT tests rotation limits)."""
    h, w = 512, 512
    img = create_rich_background(h, w, seed=202)
    truth = np.zeros((h, w), dtype=np.uint8)

    obj, alpha = draw_textured_object(size=120)
    oh, ow = obj.shape[:2]

    # Source location (unrotated)
    sx, sy = 60, 260
    roi_src = img[sy:sy+oh, sx:sx+ow]
    img[sy:sy+oh, sx:sx+ow] = np.where(alpha[:, :, None] > 0, obj, roi_src)
    truth[sy:sy+oh, sx:sx+ow] = np.where(alpha > 0, 255, 0)

    # Rotate object by 25 degrees
    M = cv2.getRotationMatrix2D((ow / 2, oh / 2), 25, 1.0)
    rot_obj = cv2.warpAffine(obj, M, (ow, oh), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rot_alpha = cv2.warpAffine(alpha, M, (ow, oh), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Target location
    tx, ty = 310, 230
    roi_tgt = img[ty:ty+oh, tx:tx+ow]
    img[ty:ty+oh, tx:tx+ow] = np.where(rot_alpha[:, :, None] > 0, rot_obj, roi_tgt)
    truth[ty:ty+oh, tx:tx+ow] = np.where(rot_alpha > 0, 255, truth[ty:ty+oh, tx:tx+ow])

    cv2.imwrite(os.path.join(out_dir, "sample2_rotation.png"), img)
    cv2.imwrite(os.path.join(out_dir, "sample2_rotation_mask.png"), truth)

def generate_sample_scaling(out_dir):
    """Case 3: Scaled Copy-Move (1.2x scale up)."""
    h, w = 512, 512
    img = create_rich_background(h, w, seed=303)
    truth = np.zeros((h, w), dtype=np.uint8)

    obj, alpha = draw_textured_object(size=90)
    oh, ow = obj.shape[:2]

    # Source location
    sx, sy = 70, 270
    roi_src = img[sy:sy+oh, sx:sx+ow]
    img[sy:sy+oh, sx:sx+ow] = np.where(alpha[:, :, None] > 0, obj, roi_src)
    truth[sy:sy+oh, sx:sx+ow] = np.where(alpha > 0, 255, 0)

    # Scale target by 1.25x
    scale = 1.25
    scaled_w, scaled_h = int(ow * scale), int(oh * scale)
    scaled_obj = cv2.resize(obj, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
    scaled_alpha = cv2.resize(alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

    tx, ty = 300, 210
    roi_tgt = img[ty:ty+scaled_h, tx:tx+scaled_w]
    img[ty:ty+scaled_h, tx:tx+scaled_w] = np.where(scaled_alpha[:, :, None] > 0, scaled_obj, roi_tgt)
    truth[ty:ty+scaled_h, tx:tx+scaled_w] = np.where(scaled_alpha > 0, 255, truth[ty:ty+scaled_h, tx:tx+scaled_w])

    cv2.imwrite(os.path.join(out_dir, "sample3_scaling.png"), img)
    cv2.imwrite(os.path.join(out_dir, "sample3_scaling_mask.png"), truth)

def generate_sample_authentic(out_dir):
    """Case 4: Negative Control (Authentic unmanipulated image with natural background)."""
    h, w = 512, 512
    img = create_rich_background(h, w, seed=404)
    obj, alpha = draw_textured_object(size=100)
    oh, ow = obj.shape[:2]

    # Only single instance (no clone)
    sx, sy = 180, 280
    roi_src = img[sy:sy+oh, sx:sx+ow]
    img[sy:sy+oh, sx:sx+ow] = np.where(alpha[:, :, None] > 0, obj, roi_src)
    truth = np.zeros((h, w), dtype=np.uint8) # No forgery

    cv2.imwrite(os.path.join(out_dir, "sample4_authentic.png"), img)
    cv2.imwrite(os.path.join(out_dir, "sample4_authentic_mask.png"), truth)

def generate_matlab_style_demo(out_dir):
    """Case 5: Exact Python port of matlab/create_demo.m."""
    np.random.seed(7)
    height, width = 320, 384
    raw = np.random.rand(height, width).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    texture = cv2.filter2D(raw, -1, kernel)
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    
    original = np.repeat((30 + 200 * texture)[:, :, None], 3, axis=2).astype(np.uint8)
    
    # Colored geometric shapes
    original[64:120, 56:136, 0] = 40  # BGR order
    original[64:120, 56:136, 1] = 60
    original[64:120, 56:136, 2] = 220
    
    patch = np.repeat((230 * texture[76:108, 68:124])[:, :, None], 3, axis=2).astype(np.uint8)
    original[76:108, 68:124] = patch
    original[128:144, 56:100, 1] = 230
    
    source_rows = slice(48, 160)
    source_cols = slice(40, 152)
    target_rows = slice(176, 288)
    target_cols = slice(216, 328)
    
    forged = original.copy()
    forged[target_rows, target_cols] = original[source_rows, source_cols]
    
    truth = np.zeros((height, width), dtype=np.uint8)
    truth[source_rows, source_cols] = 255
    truth[target_rows, target_cols] = 255
    
    cv2.imwrite(os.path.join(out_dir, "sample5_matlab_demo.png"), forged)
    cv2.imwrite(os.path.join(out_dir, "sample5_matlab_demo_mask.png"), truth)

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(out_dir, exist_ok=True)
    generate_sample_translation(out_dir)
    generate_sample_rotation(out_dir)
    generate_sample_scaling(out_dir)
    generate_sample_authentic(out_dir)
    generate_matlab_style_demo(out_dir)
    print(f"Generated 5 test pairs (image + truth mask) in: {os.path.abspath(out_dir)}")

import numpy as np
import os
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from colorsys import rgb_to_hls, hls_to_rgb

MODEL_CACHE = "trained_curves.npz"


def get_pairs_from_folders(before_folder, after_folder):
    """
    Auto-matches images from before/after folders by their numeric prefix.
    e.g. before/12_d.jpg <-> after/12_l.jpg
    """
    before_files = {}
    after_files = {}
    valid_ext = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

    for f in os.listdir(before_folder):
        if f.lower().endswith(valid_ext):
            prefix = f.split('_')[0]
            before_files[prefix] = os.path.join(before_folder, f)

    for f in os.listdir(after_folder):
        if f.lower().endswith(valid_ext):
            prefix = f.split('_')[0]
            after_files[prefix] = os.path.join(after_folder, f)

    pairs = []
    matched = sorted(set(before_files.keys()) & set(after_files.keys()))
    for prefix in matched:
        pairs.append((before_files[prefix], after_files[prefix]))

    unmatched_before = set(before_files.keys()) - set(after_files.keys())
    unmatched_after = set(after_files.keys()) - set(before_files.keys())
    if unmatched_before:
        print(f"   Warning: No match in after/ for: {[before_files[p] for p in unmatched_before]}")
    if unmatched_after:
        print(f"   Warning: No match in before/ for: {[after_files[p] for p in unmatched_after]}")

    print(f"   Found {len(pairs)} matched pairs")
    return pairs


# ==========================================
#       WHITE BALANCE CORRECTION
# ==========================================

def estimate_white_balance(img_array):
    """Estimate white balance as average R/G/B ratios (gray-world assumption)."""
    means = img_array.mean(axis=(0, 1))  # [R_mean, G_mean, B_mean]
    gray = means.mean()
    if gray == 0:
        return np.array([1.0, 1.0, 1.0])
    return means / gray  # Ratios relative to neutral gray


def normalize_white_balance(src_array, tgt_array):
    """
    Corrects white balance difference between source and target so the model
    learns the creative grade, not the WB shift from different shooting conditions.
    Returns WB-corrected target array.
    """
    src_wb = estimate_white_balance(src_array)
    tgt_wb = estimate_white_balance(tgt_array)

    # Compute the creative WB shift (intentional) vs environmental WB shift
    # We correct the environmental part: align target WB base to source WB base
    # then re-apply the creative shift
    wb_ratio = tgt_wb / np.where(src_wb > 0.01, src_wb, 0.01)

    return wb_ratio


# ==========================================
#       HSL-AWARE CURVE EXTRACTION
# ==========================================

def rgb_array_to_hsl(rgb):
    """Convert Nx3 RGB (0-255) array to Nx3 HSL (H:0-360, S:0-1, L:0-1)."""
    rgb_norm = rgb.astype(float) / 255.0
    hsl = np.zeros_like(rgb_norm)
    for i in range(len(rgb_norm)):
        r, g, b = rgb_norm[i]
        h, l, s = rgb_to_hls(r, g, b)
        hsl[i] = [h * 360, s, l]  # H:0-360, S:0-1, L:0-1
    return hsl


def build_hsl_curves(src_pixels, tgt_pixels, n_samples=50000):
    """
    Build HSL adjustment curves by learning how H, S, L each shift
    as a function of the source HSL values.
    Returns hue_shift_curve, sat_scale_curve, lum_curve (each 256-entry).
    """
    # Subsample for speed (HSL conversion is slow per-pixel)
    if len(src_pixels) > n_samples:
        idx = np.random.choice(len(src_pixels), n_samples, replace=False)
        src_sub = src_pixels[idx]
        tgt_sub = tgt_pixels[idx]
    else:
        src_sub = src_pixels
        tgt_sub = tgt_pixels

    print(f"   Building HSL curves from {len(src_sub):,} samples...")

    src_hsl = rgb_array_to_hsl(src_sub)
    tgt_hsl = rgb_array_to_hsl(tgt_sub)

    # Hue shift curve: for each source hue bin, what's the average hue shift?
    hue_bins = np.linspace(0, 360, 37)  # 36 bins of 10 degrees each
    hue_shifts = np.zeros(36)
    hue_counts = np.zeros(36)

    for i in range(36):
        lo, hi = hue_bins[i], hue_bins[i + 1]
        mask = (src_hsl[:, 0] >= lo) & (src_hsl[:, 0] < hi) & (src_hsl[:, 1] > 0.05)
        if np.sum(mask) > 10:
            # Circular mean for hue difference
            dh = tgt_hsl[mask, 0] - src_hsl[mask, 0]
            # Wrap to -180..180
            dh = (dh + 180) % 360 - 180
            hue_shifts[i] = np.mean(dh)
            hue_counts[i] = np.sum(mask)

    # Saturation scale curve: for each luminance bin, what's the sat multiplier?
    lum_bins = np.arange(256)
    sat_scales = np.ones(256)
    sat_counts = np.zeros(256)

    src_lum_int = (src_hsl[:, 2] * 255).clip(0, 255).astype(int)
    for val in range(256):
        mask = (src_lum_int == val) & (src_hsl[:, 1] > 0.02)
        if np.sum(mask) > 5:
            src_sat_mean = np.mean(src_hsl[mask, 1])
            tgt_sat_mean = np.mean(tgt_hsl[mask, 1])
            if src_sat_mean > 0.01:
                sat_scales[val] = tgt_sat_mean / src_sat_mean
                sat_counts[val] = np.sum(mask)

    # Smooth the saturation curve
    valid_sat = sat_counts > 0
    if np.sum(valid_sat) > 5:
        interp_sat = interp1d(
            lum_bins[valid_sat], sat_scales[valid_sat],
            kind="linear", fill_value="extrapolate", bounds_error=False
        )
        sat_scales = interp_sat(lum_bins).clip(0.0, 3.0)
        try:
            sat_scales = savgol_filter(sat_scales, window_length=15, polyorder=2)
            sat_scales = sat_scales.clip(0.0, 3.0)
        except ValueError:
            pass

    return hue_shifts, hue_counts, sat_scales


# ==========================================
#    SHADOW / MIDTONE / HIGHLIGHT SPLIT
# ==========================================

def build_tonal_curves(src_pixels, tgt_pixels):
    """
    Builds separate RGB curves for shadows, midtones, and highlights.
    Shadow: L < 85 (0-1/3), Midtone: 85-170 (1/3-2/3), Highlight: > 170 (2/3-1)
    Returns dict with 'shadow', 'midtone', 'highlight' curves (each 3x256).
    """
    # Compute luminance for each pixel
    src_lum = (0.299 * src_pixels[:, 0] + 0.587 * src_pixels[:, 1] + 0.114 * src_pixels[:, 2])

    zones = {
        'shadow':    (0, 85),
        'midtone':   (85, 170),
        'highlight': (170, 256),
    }

    tonal_curves = {}
    x_bins = np.arange(256)

    for zone_name, (lo, hi) in zones.items():
        zone_mask = (src_lum >= lo) & (src_lum < hi)
        zone_src = src_pixels[zone_mask]
        zone_tgt = tgt_pixels[zone_mask]

        if len(zone_src) < 100:
            print(f"   {zone_name}: too few pixels ({len(zone_src)}), using global curve")
            tonal_curves[zone_name] = None
            continue

        curves = []
        for ch in range(3):
            y_means = np.full(256, np.nan)
            for val in x_bins:
                mask = zone_src[:, ch] == val
                if np.sum(mask) > 0:
                    y_means[val] = np.mean(zone_tgt[mask, ch])

            valid = ~np.isnan(y_means)
            if np.sum(valid) < 3:
                curves.append(x_bins.astype(np.float64))
                continue

            interp = interp1d(
                x_bins[valid], y_means[valid],
                kind="linear", fill_value="extrapolate", bounds_error=False
            )
            curve = interp(x_bins)
            try:
                curve = savgol_filter(curve, window_length=11, polyorder=2)
            except ValueError:
                pass
            curves.append(curve.clip(0, 255))

        tonal_curves[zone_name] = np.array(curves)  # 3x256
        print(f"   {zone_name}: {len(zone_src):,} pixels")

    return tonal_curves


# ==========================================
#       MAIN TRAINING FUNCTION
# ==========================================

def extract_curves_from_pairs(pairs):
    """
    Learns color curves from multiple before/after pairs with:
    - Per-channel RGB curves (weighted + smoothed)
    - HSL-aware hue/saturation adjustments
    - Shadow/Midtone/Highlight tonal split
    - White balance normalization
    """
    all_src_pixels = []
    all_tgt_pixels = []
    wb_ratios = []

    for i, (raw_path, edited_path) in enumerate(pairs, 1):
        print(f"   Loading pair {i}/{len(pairs)}: {raw_path} -> {edited_path}")

        raw_ref = np.array(Image.open(raw_path).convert("RGB"))
        edited_ref = np.array(Image.open(edited_path).convert("RGB"))

        if raw_ref.shape != edited_ref.shape:
            edited_ref = np.array(
                Image.open(edited_path).convert("RGB")
                .resize((raw_ref.shape[1], raw_ref.shape[0]), Image.Resampling.LANCZOS)
            )

        # White balance analysis
        wb = normalize_white_balance(raw_ref, edited_ref)
        wb_ratios.append(wb)

        all_src_pixels.append(raw_ref.reshape(-1, 3))
        all_tgt_pixels.append(edited_ref.reshape(-1, 3))

    all_src = np.concatenate(all_src_pixels)
    all_tgt = np.concatenate(all_tgt_pixels)
    avg_wb = np.mean(wb_ratios, axis=0)

    total_pixels = len(all_src)
    print(f"   Total training pixels: {total_pixels:,} (from {len(pairs)} pairs)")
    print(f"   Avg WB shift: R={avg_wb[0]:.3f} G={avg_wb[1]:.3f} B={avg_wb[2]:.3f}")

    # --- Per-channel RGB curves (with weighting + smoothing) ---
    print("\n   Building per-channel RGB curves...")
    rgb_luts = []
    channel_names = ["Red", "Green", "Blue"]
    x_bins = np.arange(256)

    for ch in range(3):
        src_flat = all_src[:, ch]
        tgt_flat = all_tgt[:, ch]

        y_means = np.full(256, np.nan)
        y_counts = np.zeros(256, dtype=int)
        for val in x_bins:
            mask = src_flat == val
            count = np.sum(mask)
            if count > 0:
                y_means[val] = np.mean(tgt_flat[mask])
                y_counts[val] = count

        coverage = np.sum(y_counts > 0)
        print(f"   {channel_names[ch]}: {coverage}/256 values covered")

        valid_mask = ~np.isnan(y_means)
        if not valid_mask.any():
            print(f"   ERROR: {channel_names[ch]} channel has no data!")
            return None

        # Weighted smoothing for low-confidence values
        y_weighted = y_means.copy()
        for idx in np.where(valid_mask)[0]:
            if y_counts[idx] < 100:
                neighbors = y_means[max(0, idx-2):min(256, idx+3)]
                valid_n = neighbors[~np.isnan(neighbors)]
                if len(valid_n) > 0:
                    y_weighted[idx] = np.mean(valid_n)

        interp = interp1d(
            x_bins[valid_mask], y_weighted[valid_mask],
            kind="linear", fill_value="extrapolate", bounds_error=False,
        )
        full_lut = interp(np.arange(256))

        try:
            full_lut = savgol_filter(full_lut, window_length=11, polyorder=2)
        except ValueError:
            full_lut = savgol_filter(full_lut, window_length=5, polyorder=1)

        rgb_luts.append(full_lut.clip(0, 255).astype(np.uint8))

    # --- HSL curves ---
    print("\n   Building HSL curves...")
    hue_shifts, hue_counts, sat_scales = build_hsl_curves(all_src, all_tgt)

    # --- Tonal zone curves ---
    print("\n   Building shadow/midtone/highlight curves...")
    tonal_curves = build_tonal_curves(all_src, all_tgt)

    return {
        'rgb_luts': rgb_luts,           # 3 x 256 uint8
        'hue_shifts': hue_shifts,       # 36 floats
        'hue_counts': hue_counts,       # 36 floats
        'sat_scales': sat_scales,       # 256 floats
        'tonal_curves': tonal_curves,   # dict of 3x256 arrays
        'wb_ratio': avg_wb,             # 3 floats
    }


# ==========================================
#          SAVE / LOAD CACHE
# ==========================================

def save_trained_curves(model, filename=None):
    """Saves full trained model to disk."""
    filename = filename or MODEL_CACHE
    save_dict = {
        'r': model['rgb_luts'][0],
        'g': model['rgb_luts'][1],
        'b': model['rgb_luts'][2],
        'hue_shifts': model['hue_shifts'],
        'hue_counts': model['hue_counts'],
        'sat_scales': model['sat_scales'],
        'wb_ratio': model['wb_ratio'],
    }
    # Save tonal curves
    for zone in ['shadow', 'midtone', 'highlight']:
        if model['tonal_curves'][zone] is not None:
            save_dict[f'tonal_{zone}'] = model['tonal_curves'][zone]

    np.savez(filename, **save_dict)
    print(f"   Model saved to {filename}")


def load_trained_curves(filename=None):
    """Loads cached model if the file exists."""
    filename = filename or MODEL_CACHE
    if not os.path.exists(filename):
        return None

    data = np.load(filename)
    print(f"   Loaded cached model from {filename} (skipping training)")

    tonal_curves = {}
    for zone in ['shadow', 'midtone', 'highlight']:
        key = f'tonal_{zone}'
        tonal_curves[zone] = data[key] if key in data else None

    return {
        'rgb_luts': [data['r'], data['g'], data['b']],
        'hue_shifts': data['hue_shifts'],
        'hue_counts': data['hue_counts'],
        'sat_scales': data['sat_scales'],
        'tonal_curves': tonal_curves,
        'wb_ratio': data['wb_ratio'],
    }


# ==========================================
#           APPLY GRADE TO IMAGE
# ==========================================

def apply_grade(img_array, model):
    """
    Applies the full color grade to an image array:
    1. Tonal-zone-aware RGB curves (shadow/mid/highlight)
    2. HSL adjustments (hue shifts + saturation scaling)
    3. White balance correction baked in
    """
    rgb_luts = model['rgb_luts']
    hue_shifts = model['hue_shifts']
    hue_counts = model['hue_counts']
    sat_scales = model['sat_scales']
    tonal = model['tonal_curves']

    h, w, _ = img_array.shape
    result = img_array.copy().astype(np.float64)

    # Step 1: Apply tonal-zone curves
    luminance = (0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2])

    # Shadow zone
    shadow_mask = luminance < 85
    midtone_mask = (luminance >= 85) & (luminance < 170)
    highlight_mask = luminance >= 170

    # Smooth blending weights (avoid hard transitions between zones)
    # Shadow blend: 1 at L=0, fades to 0 at L=85-127
    # Highlight blend: 0 at L=128-170, fades to 1 at L=255
    shadow_weight = np.clip((100 - luminance) / 50, 0, 1)
    highlight_weight = np.clip((luminance - 155) / 50, 0, 1)
    midtone_weight = 1.0 - shadow_weight - highlight_weight
    midtone_weight = np.clip(midtone_weight, 0, 1)

    for ch in range(3):
        global_mapped = rgb_luts[ch][img_array[:, :, ch]]
        result_ch = global_mapped.astype(np.float64)

        # Blend in tonal curves where available
        if tonal['shadow'] is not None:
            s_curve = tonal['shadow'][ch]
            shadow_mapped = s_curve[img_array[:, :, ch].clip(0, 255).astype(int)]
            result_ch = result_ch * (1 - shadow_weight) + shadow_mapped * shadow_weight

        if tonal['highlight'] is not None:
            h_curve = tonal['highlight'][ch]
            high_mapped = h_curve[img_array[:, :, ch].clip(0, 255).astype(int)]
            result_ch = result_ch * (1 - highlight_weight) + high_mapped * highlight_weight

        result[:, :, ch] = result_ch

    # Step 2: HSL adjustments (hue shift + saturation scaling)
    result_uint8 = result.clip(0, 255).astype(np.uint8)
    result_flat = result_uint8.reshape(-1, 3)

    # Vectorized HSL conversion using PIL
    hsl_img = Image.fromarray(result_uint8).convert('HSV')
    hsv_array = np.array(hsl_img)  # H:0-255, S:0-255, V:0-255

    # Apply hue shifts (map H from 0-255 to 0-360 bin index)
    h_float = hsv_array[:, :, 0].astype(float) / 255.0 * 360.0
    bin_indices = (h_float / 10.0).astype(int).clip(0, 35)

    for i in range(36):
        if abs(hue_shifts[i]) > 0.5 and hue_counts[i] > 50:
            mask = bin_indices == i
            h_float[mask] = (h_float[mask] + hue_shifts[i]) % 360

    hsv_array[:, :, 0] = (h_float / 360.0 * 255.0).clip(0, 255).astype(np.uint8)

    # Apply saturation scaling based on luminance
    lum_for_sat = (result[:, :, 0] * 0.299 + result[:, :, 1] * 0.587 + result[:, :, 2] * 0.114)
    lum_idx = lum_for_sat.clip(0, 255).astype(int)
    sat_multiplier = sat_scales[lum_idx]

    sat_float = hsv_array[:, :, 1].astype(float) * sat_multiplier
    hsv_array[:, :, 1] = sat_float.clip(0, 255).astype(np.uint8)

    # Convert back to RGB
    result_img = Image.fromarray(hsv_array, 'HSV').convert('RGB')
    final = np.array(result_img)

    return final


# ==========================================
#           CUBE FILE EXPORT
# ==========================================

def save_cube_file(model, filename, lut_size=64):
    """
    Exports the full grade as a standard .cube 3D LUT file.
    The .cube bakes in RGB curves + HSL adjustments + tonal zones.
    """
    print(f"   Saving {lut_size}x{lut_size}x{lut_size} .cube file: {filename}")

    domain = np.linspace(0, 255, lut_size).astype(int)

    with open(filename, "w") as f:
        f.write(f'TITLE "Color_Stealer_Pro"\n')
        f.write(f"LUT_3D_SIZE {lut_size}\n\n")

        for b_val in domain:
            for g_val in domain:
                for r_val in domain:
                    # Create a 1x1 pixel image with this color
                    pixel = np.array([[[r_val, g_val, b_val]]], dtype=np.uint8)
                    graded = apply_grade(pixel, model)
                    nr = graded[0, 0, 0] / 255.0
                    ng = graded[0, 0, 1] / 255.0
                    nb = graded[0, 0, 2] / 255.0
                    f.write(f"{nr:.6f} {ng:.6f} {nb:.6f}\n")

    print(f"   .cube file saved!")


def save_cube_file_fast(model, filename, lut_size=64):
    """
    Fast .cube export using only RGB curves (no per-pixel HSL processing).
    Use this for quick exports; use save_cube_file() for full accuracy.
    """
    print(f"   Saving {lut_size}x{lut_size}x{lut_size} .cube file (fast): {filename}")
    luts = model['rgb_luts']

    with open(filename, "w") as f:
        f.write(f'TITLE "Color_Stealer_Fast"\n')
        f.write(f"LUT_3D_SIZE {lut_size}\n\n")

        domain = np.linspace(0, 255, lut_size).astype(int)
        for b_val in domain:
            for g_val in domain:
                for r_val in domain:
                    new_r = luts[0][r_val] / 255.0
                    new_g = luts[1][g_val] / 255.0
                    new_b = luts[2][b_val] / 255.0
                    f.write(f"{new_r:.6f} {new_g:.6f} {new_b:.6f}\n")

    print(f"   .cube file saved!")


# ==========================================
#          BATCH APPLY TO FOLDER
# ==========================================

def apply_to_folder(model, target_folder, output_folder):
    """Applies the full learned grade to every image in a folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [
        f for f in os.listdir(target_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ]

    if not files:
        print(f"   No images found in {target_folder}")
        return

    print(f"   Processing {len(files)} images...")
    for filename in files:
        img_path = os.path.join(target_folder, filename)
        img = Image.open(img_path)
        img = img.convert("RGB")
        img_array = np.array(img)

        # Apply full grade
        result_array = apply_grade(img_array, model)

        result_img = Image.fromarray(result_array)

        # Preserve EXIF data if available
        original_img = Image.open(img_path)
        exif_data = original_img.info.get("exif", None)

        name, ext = os.path.splitext(filename)
        save_path = os.path.join(output_folder, f"Graded_{filename}")

        save_kwargs = {}
        if exif_data:
            save_kwargs["exif"] = exif_data

        if ext.lower() in (".jpg", ".jpeg"):
            save_kwargs["quality"] = 100
            save_kwargs["subsampling"] = 0
        elif ext.lower() == ".png":
            save_kwargs["compress_level"] = 1
        elif ext.lower() in (".tif", ".tiff"):
            save_kwargs["compression"] = "tiff_lzw"

        result_img.save(save_path, **save_kwargs)
        print(f"   Done: {filename} ({img_array.shape[1]}x{img_array.shape[0]})")
    print(f"   All images saved at original resolution & maximum quality.")


# ==========================================
#              EXECUTION SECTION
# ==========================================

# Training folders — put before/after images here
BEFORE_FOLDER = "./before"
AFTER_FOLDER = "./after"

# Auto-pair images from the two folders
pairs = get_pairs_from_folders(BEFORE_FOLDER, AFTER_FOLDER)

if not pairs:
    print("ERROR: No matching pairs found. Make sure before/ and after/ folders")
    print("       have images with matching prefixes (e.g. 12_d.jpg <-> 12_l.jpg)")
else:
    # 1. Load cached model or train from scratch
    print("=" * 50)
    print("PHASE 1: Learning Color Grade")
    print("=" * 50)
    model = load_trained_curves()
    if model is None:
        model = extract_curves_from_pairs(pairs)
        if model:
            save_trained_curves(model)

    if model:
        # Need hue_shifts/hue_counts accessible for apply_grade
        hue_shifts = model['hue_shifts']
        hue_counts = model['hue_counts']

        # 2. Export .cube file only if it doesn't exist
        if not os.path.exists("My_Style.cube"):
            print("\n" + "=" * 50)
            print("PHASE 2: Exporting .cube LUT (fast mode)")
            print("=" * 50)
            save_cube_file_fast(model, "My_Style.cube", lut_size=64)
        else:
            print("\n   My_Style.cube already exists, skipping export.")

        # 3. Batch-apply to raw photos (full grade with HSL + tonal zones)
        print("\n" + "=" * 50)
        print("PHASE 3: Batch Processing (full grade)")
        print("=" * 50)
        apply_to_folder(model, "./Raw_Photos", "./Finished_Photos")

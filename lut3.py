import numpy as np
import os
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

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


def estimate_white_balance(img_array):
    """Estimate white balance as average R/G/B ratios."""
    means = img_array.mean(axis=(0, 1))
    gray = means.mean()
    if gray == 0:
        return np.array([1.0, 1.0, 1.0])
    return means / gray  # Ratios relative to neutral gray


def extract_curves_from_pairs(pairs):
    """
    Learns per-channel RGB curves + shadow/midtone/highlight zones.
    Simple, stable approach with weighted averaging and smoothing.
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
        wb = estimate_white_balance(raw_ref)
        wb_ratios.append(wb)

        all_src_pixels.append(raw_ref.reshape(-1, 3))
        all_tgt_pixels.append(edited_ref.reshape(-1, 3))

    all_src = np.concatenate(all_src_pixels)
    all_tgt = np.concatenate(all_tgt_pixels)
    avg_wb = np.mean(wb_ratios, axis=0)

    total_pixels = len(all_src)
    print(f"   Total training pixels: {total_pixels:,} (from {len(pairs)} pairs)")
    print(f"   Avg WB shift: R={avg_wb[0]:.3f} G={avg_wb[1]:.3f} B={avg_wb[2]:.3f}")

    # --- Build per-channel RGB curves ---
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

    # --- Shadow/Midtone/Highlight split ---
    print("\n   Building shadow/midtone/highlight curves...")
    tonal_curves = build_tonal_curves(all_src, all_tgt)

    return {
        'rgb_luts': rgb_luts,
        'tonal_curves': tonal_curves,
        'wb_ratio': avg_wb,
    }


def build_tonal_curves(src_pixels, tgt_pixels):
    """
    Builds separate RGB curves for shadows, midtones, and highlights.
    Shadow: L < 85, Midtone: 85-170, Highlight: > 170
    """
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
            print(f"   {zone_name}: too few pixels ({len(zone_src)}), skipping")
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

        tonal_curves[zone_name] = np.array(curves)
        print(f"   {zone_name}: {len(zone_src):,} pixels")

    return tonal_curves


def save_trained_curves(model, filename=None):
    """Saves trained model to disk."""
    filename = filename or MODEL_CACHE
    save_dict = {
        'r': model['rgb_luts'][0],
        'g': model['rgb_luts'][1],
        'b': model['rgb_luts'][2],
        'wb_ratio': model['wb_ratio'],
    }
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
        'tonal_curves': tonal_curves,
        'wb_ratio': data['wb_ratio'],
    }


def apply_grade(img_array, model):
    """
    Applies per-channel RGB curves with tonal zone blending.
    Simple, stable approach without HSL artifacts.
    """
    rgb_luts = model['rgb_luts']
    tonal = model['tonal_curves']

    h, w, _ = img_array.shape
    result = img_array.copy().astype(np.float64)

    # Compute luminance for tonal zone weighting
    luminance = (0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2])

    # Smooth blending weights for tonal zones
    shadow_weight = np.clip((100 - luminance) / 50, 0, 1)
    highlight_weight = np.clip((luminance - 155) / 50, 0, 1)
    midtone_weight = 1.0 - shadow_weight - highlight_weight
    midtone_weight = np.clip(midtone_weight, 0, 1)

    # Apply curves per-channel
    for ch in range(3):
        # Global RGB curve (always applied)
        global_mapped = rgb_luts[ch][img_array[:, :, ch]]
        result_ch = global_mapped.astype(np.float64)

        # Optionally blend in tonal zone curves if available
        if tonal['shadow'] is not None:
            s_curve = tonal['shadow'][ch]
            shadow_mapped = s_curve[img_array[:, :, ch].clip(0, 255).astype(int)]
            result_ch = result_ch * (1 - shadow_weight * 0.5) + shadow_mapped * shadow_weight * 0.5

        if tonal['highlight'] is not None:
            h_curve = tonal['highlight'][ch]
            high_mapped = h_curve[img_array[:, :, ch].clip(0, 255).astype(int)]
            result_ch = result_ch * (1 - highlight_weight * 0.5) + high_mapped * highlight_weight * 0.5

        result[:, :, ch] = result_ch

    return result.clip(0, 255).astype(np.uint8)


def save_cube_file(model, filename, lut_size=64):
    """Exports per-channel RGB curves as a standard .cube 3D LUT file."""
    print(f"   Saving {lut_size}x{lut_size}x{lut_size} .cube file: {filename}")
    luts = model['rgb_luts']

    with open(filename, "w") as f:
        f.write(f'TITLE "Color_Stealer"\n')
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


def apply_to_folder(model, target_folder, output_folder):
    """Applies the learned grade to every image in a folder."""
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

        # Apply grade
        result_array = apply_grade(img_array, model)

        result_img = Image.fromarray(result_array)

        # Preserve EXIF if available
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
        # 2. Export .cube file only if it doesn't exist
        if not os.path.exists("My_Style.cube"):
            print("\n" + "=" * 50)
            print("PHASE 2: Exporting .cube LUT")
            print("=" * 50)
            save_cube_file(model, "My_Style.cube", lut_size=64)
        else:
            print("\n   My_Style.cube already exists, skipping export.")

        # 3. Batch-apply to raw photos
        print("\n" + "=" * 50)
        print("PHASE 3: Batch Processing")
        print("=" * 50)
        apply_to_folder(model, "./Raw_Photos", "./Finished_Photos")

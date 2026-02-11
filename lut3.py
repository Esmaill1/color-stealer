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

    # Index before files by numeric prefix
    for f in os.listdir(before_folder):
        if f.lower().endswith(valid_ext):
            prefix = f.split('_')[0]  # e.g. "12" from "12_d.jpg"
            before_files[prefix] = os.path.join(before_folder, f)

    # Index after files by numeric prefix
    for f in os.listdir(after_folder):
        if f.lower().endswith(valid_ext):
            prefix = f.split('_')[0]
            after_files[prefix] = os.path.join(after_folder, f)

    # Match by prefix
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


def extract_curves_from_pairs(pairs):
    """
    Learns per-channel color curves from MULTIPLE before/after image pairs.
    More pairs = better color space coverage = more accurate LUT.

    Args:
        pairs: list of tuples [(raw_path, edited_path), ...]

    Returns:
        luts: list of 3 numpy arrays (R, G, B), each mapping 0-255 -> 0-255
    """
    # Pool ALL pixel data from every pair
    channel_src = [[], [], []]  # R, G, B source pixels
    channel_tgt = [[], [], []]  # R, G, B target pixels

    for i, (raw_path, edited_path) in enumerate(pairs, 1):
        print(f"   Loading pair {i}/{len(pairs)}: {raw_path} -> {edited_path}")

        raw_ref = np.array(Image.open(raw_path).convert("RGB"))
        edited_ref = np.array(Image.open(edited_path).convert("RGB"))

        if raw_ref.shape != edited_ref.shape:
            # Auto-resize target to match source if dimensions differ
            edited_ref = np.array(
                Image.open(edited_path)
                .convert("RGB")
                .resize((raw_ref.shape[1], raw_ref.shape[0]), Image.Resampling.LANCZOS)
            )

        for ch in range(3):
            channel_src[ch].append(raw_ref[:, :, ch].flatten())
            channel_tgt[ch].append(edited_ref[:, :, ch].flatten())

    # Concatenate all pixel data from all pairs
    for ch in range(3):
        channel_src[ch] = np.concatenate(channel_src[ch])
        channel_tgt[ch] = np.concatenate(channel_tgt[ch])

    total_pixels = len(channel_src[0])
    print(f"   Total training pixels: {total_pixels:,} (from {len(pairs)} pairs)")

    # Build per-channel LUTs from the pooled data
    luts = []
    channel_names = ["Red", "Green", "Blue"]
    x_bins = np.arange(256)

    for ch in range(3):
        src_flat = channel_src[ch]
        tgt_flat = channel_tgt[ch]

        y_means = []
        y_counts = []
        for val in x_bins:
            mask = src_flat == val
            count = np.sum(mask)
            if count > 0:
                y_means.append(np.mean(tgt_flat[mask]))
                y_counts.append(count)
            else:
                y_means.append(np.nan)
                y_counts.append(0)

        y_means = np.array(y_means)
        y_counts = np.array(y_counts)

        # Report coverage
        coverage = np.sum(y_counts > 0)
        print(f"   {channel_names[ch]}: {coverage}/256 values covered")

        valid_mask = ~np.isnan(y_means)
        if not valid_mask.any():
            print(f"   ERROR: {channel_names[ch]} channel has no data!")
            return None

        # Normalize counts for weighting (log scale for stability)
        # Rare pixels (few counts) have lower weight; common pixels have higher weight
        max_count = np.max(y_counts[y_counts > 0])
        min_count = np.min(y_counts[y_counts > 0])
        
        # Use log weighting to compress the range but still differentiate
        normalized_counts = np.zeros_like(y_counts, dtype=float)
        normalized_counts[y_counts > 0] = np.log1p(y_counts[y_counts > 0] / max_count * 100)
        normalized_counts = normalized_counts / np.max(normalized_counts)
        
        # Weighted interpolation: values with higher confidence are more influential
        y_means_weighted = y_means.copy()
        for i in np.where(valid_mask)[0]:
            if y_counts[i] < 100:  # Low-confidence values
                # Smooth them towards neighbors
                neighbors = y_means[max(0, i-2):min(256, i+3)]
                valid_neighbors = neighbors[~np.isnan(neighbors)]
                if len(valid_neighbors) > 0:
                    y_means_weighted[i] = np.mean(valid_neighbors)
        
        # Primary interpolation for missing values
        interpolator = interp1d(
            x_bins[valid_mask],
            y_means_weighted[valid_mask],
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )

        full_lut = interpolator(np.arange(256))
        
        # Apply Savitzky-Golay smoothing for professional, clean curves
        # window_length=11 for gentle smoothing, polyorder=2 for polynomial fit
        try:
            full_lut = savgol_filter(full_lut, window_length=11, polyorder=2)
        except ValueError:
            # If dataset is too small, fall back to simpler smoothing
            full_lut = savgol_filter(full_lut, window_length=5, polyorder=1)
        
        full_lut = full_lut.clip(0, 255).astype(np.uint8)
        luts.append(full_lut)

    return luts


def save_trained_curves(luts, filename=None):
    """Saves trained LUT curves to disk."""
    filename = filename or MODEL_CACHE
    np.savez(filename, r=luts[0], g=luts[1], b=luts[2])
    print(f"   Curves saved to {filename}")


def load_trained_curves(filename=None):
    """
    Loads cached curves if the file exists.
    Returns the luts if found, None otherwise.
    """
    filename = filename or MODEL_CACHE
    if not os.path.exists(filename):
        return None

    data = np.load(filename)
    print(f"   Loaded cached curves from {filename} (skipping training)")
    return [data["r"], data["g"], data["b"]]


def save_cube_file(luts, filename, lut_size=64):
    """
    Exports the per-channel curves as a standard .cube 3D LUT file.
    Compatible with Photoshop, Premiere, DaVinci Resolve, Lightroom, etc.
    """
    print(f"   Saving {lut_size}x{lut_size}x{lut_size} .cube file: {filename}")

    with open(filename, "w") as f:
        f.write(f'TITLE "Multi_Pair_Style_Match"\n')
        f.write(f"LUT_3D_SIZE {lut_size}\n\n")

        # Generate the 3D grid and apply curves
        domain = np.linspace(0, 255, lut_size).astype(int)
        for b_val in domain:
            for g_val in domain:
                for r_val in domain:
                    new_r = luts[0][r_val] / 255.0
                    new_g = luts[1][g_val] / 255.0
                    new_b = luts[2][b_val] / 255.0
                    f.write(f"{new_r:.6f} {new_g:.6f} {new_b:.6f}\n")

    print(f"   .cube file saved!")


def apply_to_folder(luts, target_folder, output_folder):
    """Applies the learned curves to every image in a folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [
        f
        for f in os.listdir(target_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ]

    if not files:
        print(f"   No images found in {target_folder}")
        return

    print(f"   Processing {len(files)} images...")
    for filename in files:
        img_path = os.path.join(target_folder, filename)
        img = Image.open(img_path)
        original_format = img.format  # Preserve original format info
        img = img.convert("RGB")
        img_array = np.array(img)

        # Apply LUT per channel (instant — just an array lookup)
        result_array = img_array.copy()
        for ch in range(3):
            result_array[:, :, ch] = luts[ch][img_array[:, :, ch]]

        result_img = Image.fromarray(result_array)
        
        # Preserve EXIF data if available
        exif_data = img.info.get("exif", None)
        
        # Save at full quality, matching input format
        name, ext = os.path.splitext(filename)
        save_path = os.path.join(output_folder, f"Graded_{filename}")
        
        save_kwargs = {}
        if exif_data:
            save_kwargs["exif"] = exif_data
        
        if ext.lower() in (".jpg", ".jpeg"):
            save_kwargs["quality"] = 100
            save_kwargs["subsampling"] = 0  # 4:4:4 chroma — no color downsampling
        elif ext.lower() == ".png":
            save_kwargs["compress_level"] = 1  # Fast, lossless
        elif ext.lower() in (".tif", ".tiff"):
            save_kwargs["compression"] = "tiff_lzw"  # Lossless compression
        
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
    # 1. Load cached curves or train from scratch
    print("=" * 50)
    print("PHASE 1: Learning Color Grade")
    print("=" * 50)
    luts = load_trained_curves()
    if luts is None:
        luts = extract_curves_from_pairs(pairs)
        if luts:
            save_trained_curves(luts)

    if luts:
        # 2. Export .cube file only if it doesn't exist
        if not os.path.exists("My_Style.cube"):
            print("\n" + "=" * 50)
            print("PHASE 2: Exporting .cube LUT")
            print("=" * 50)
            save_cube_file(luts, "My_Style.cube", lut_size=64)
        else:
            print("\n   My_Style.cube already exists, skipping export.")

        # 3. Batch-apply to raw photos
        print("\n" + "=" * 50)
        print("PHASE 3: Batch Processing")
        print("=" * 50)
        apply_to_folder(luts, "./Raw_Photos", "./Finished_Photos")

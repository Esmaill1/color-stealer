import numpy as np
import os
from PIL import Image
from scipy.interpolate import RegularGridInterpolator, griddata, NearestNDInterpolator
from scipy.ndimage import gaussian_filter

MODEL_CACHE = "trained_lut_3d.npz"
LUT_3D_SIZE = 33  # Standard precision for 3D LUTs


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


def extract_3d_lut_from_pairs(pairs, lut_size=LUT_3D_SIZE):
    """
    Learns a full 3D LUT (R,G,B -> R',G',B') by binning pixel transformations
    into a 3D cubic grid and interpolating missing values.
    Best for 'print correction' or complex color casts.
    """
    print(f"   Initializing 3D LUT extraction (Grid Size: {lut_size}x{lut_size}x{lut_size})...")
    
    # We build a sparse accumulation grid first
    # sum_grid accumulates the TARGET colors for pixels falling in that bin
    sum_grid = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.float64)
    count_grid = np.zeros((lut_size, lut_size, lut_size), dtype=np.float64)
    
    for i, (raw_path, edited_path) in enumerate(pairs, 1):
        print(f"   Sampling pair {i}/{len(pairs)}: {os.path.basename(raw_path)}")
        
        raw_ref = np.array(Image.open(raw_path).convert("RGB"))
        edited_ref = np.array(Image.open(edited_path).convert("RGB"))

        if raw_ref.shape != edited_ref.shape:
            # Resize target to match source
            edited_ref = np.array(
                Image.open(edited_path).convert("RGB")
                .resize((raw_ref.shape[1], raw_ref.shape[0]), Image.Resampling.LANCZOS)
            )

        # Flatten arrays
        src_flat = raw_ref.reshape(-1, 3)
        tgt_flat = edited_ref.reshape(-1, 3)
        
        # --- QUANTIZATION ---
        # Map source RGB (0-255) to Grid Indices (0 to lut_size-1)
        indices = (src_flat / 255.0 * (lut_size - 1)).round().astype(int)
        indices = np.clip(indices, 0, lut_size - 1)
        
        # Accumulate data into the grid
        # np.add.at allows fast unbuffered accumulation
        idx_tuple = (indices[:, 0], indices[:, 1], indices[:, 2])
        np.add.at(sum_grid, idx_tuple, tgt_flat)
        np.add.at(count_grid, idx_tuple, 1)

    print("   Aggregating grid data and filling gaps...")
    
    # 1. Calculate Average Target Color for inhabited bins
    valid_mask = count_grid > 0
    raw_lut = np.zeros_like(sum_grid)
    raw_lut[valid_mask] = sum_grid[valid_mask] / count_grid[valid_mask][:, None]
    
    coverage_pct = np.sum(valid_mask) / valid_mask.size * 100
    print(f"   Grid Coverage: {coverage_pct:.2f}% (active bins)")

    # 2. Interpolate empty bins (this generalizes the behavior)
    # We treat valid bins as 'control points' and interpolate the rest.
    grid_points = np.argwhere(valid_mask)  # Shape (N, 3) - indices
    valid_values = raw_lut[valid_mask]     # Shape (N, 3) - target RGB
    
    # Indices of ALL points in the grid (target coordinates)
    # For a 33^3 grid, this is ~36k points.
    all_points = np.indices((lut_size, lut_size, lut_size)).reshape(3, -1).T
    
    filled_lut = np.zeros((lut_size, lut_size, lut_size, 3))
    
    # Perform interpolation per channel (R, G, B)
    for ch in range(3):
        print(f"     Interpolating Channel {ch}...")
        
        values_ch = valid_values[:, ch]
        
        # 'linear' is good inside the hull of data points
        # 'nearest' is used to fill data OUTSIDE the hull (extrapolation)
        
        # Step A: Nearest neighbor (fills everything, robust fallback)
        nn = NearestNDInterpolator(grid_points, values_ch)
        channel_full = nn(all_points)
        
        # Step B: Linear (smoother, better quality where data exists)
        # griddata returns NaN where it can't interpolate
        lin = griddata(grid_points, values_ch, all_points, method='linear')
        
        # Combine: Use linear where possible, nearest elsewhere
        mask_nan = np.isnan(lin)
        lin[mask_nan] = channel_full[mask_nan]
        
        filled_lut[..., ch] = lin.reshape((lut_size, lut_size, lut_size))

    # 3. Smooth the LUT to prevent banding/noise
    print("   Smoothing final 3D LUT (sigma=0.6)...")
    # Gaussian smoothing over the voxel grid
    smoothed_lut = np.zeros_like(filled_lut)
    for ch in range(3):
        smoothed_lut[..., ch] = gaussian_filter(filled_lut[..., ch], sigma=0.6)

    return smoothed_lut.clip(0, 255)


def save_trained_lut(lut_array, filename=None):
    """Saves trained 3D LUT to disk."""
    filename = filename or MODEL_CACHE
    np.savez(filename, lut_3d=lut_array)
    print(f"   Model saved to {filename}")


def load_trained_lut(filename=None):
    """Loads cached model if the file exists."""
    filename = filename or MODEL_CACHE
    if not os.path.exists(filename):
        return None

    data = np.load(filename)
    print(f"   Loaded cached model from {filename}")
    if 'lut_3d' in data:
        return data['lut_3d']
    return None


def apply_grade_3d(img_array, lut_3d):
    """
    Applies the learned 3D LUT to an image using trilinear interpolation.
    """
    lut_size = lut_3d.shape[0]
    
    # Setup interpolator
    # Grid coordinates range from 0 to 255
    x = np.linspace(0, 255, lut_size)
    y = np.linspace(0, 255, lut_size)
    z = np.linspace(0, 255, lut_size)
    
    # Create interpolators for R, G, B output channels
    # Input is RGB point -> Output is R (or G or B)
    interp_r = RegularGridInterpolator((x, y, z), lut_3d[:, :, :, 0], bounds_error=False, fill_value=None)
    interp_g = RegularGridInterpolator((x, y, z), lut_3d[:, :, :, 1], bounds_error=False, fill_value=None)
    interp_b = RegularGridInterpolator((x, y, z), lut_3d[:, :, :, 2], bounds_error=False, fill_value=None)
    
    # Flatten image for processing
    flat_img = img_array.reshape(-1, 3).astype(np.float64)
    
    # Apply interpolation
    # Note: RegularGridInterpolator expects (x_idx, y_idx, z_idx) which corresponds to (R, G, B)
    # assuming we built the LUT as [R][G][B]
    
    new_r = interp_r(flat_img)
    new_g = interp_g(flat_img)
    new_b = interp_b(flat_img)
    
    result = np.stack([new_r, new_g, new_b], axis=-1)
    
    return result.reshape(img_array.shape).clip(0, 255).astype(np.uint8)


def save_cube_file(lut_3d, filename):
    """Exports the 3D numpy array as a standard .cube file."""
    lut_size = lut_3d.shape[0]
    print(f"   Saving {lut_size}x{lut_size}x{lut_size} .cube file: {filename}")

    with open(filename, "w") as f:
        f.write(f'TITLE "Print_Correction_AI"\n')
        f.write(f'DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write(f'DOMAIN_MAX 1.0 1.0 1.0\n')
        f.write(f"LUT_3D_SIZE {lut_size}\n\n")

        # Cube format expects nested loops: R outer, G middle, B inner ? 
        # Actually Adobe spec: "The lines .. are in order of Red changes fastest, then Green, then Blue"
        # i.e. loop B, then G, then R
        
        # Our LUT is indexed [r_idx, g_idx, b_idx]
        
        for b_idx in range(lut_size):
            for g_idx in range(lut_size):
                for r_idx in range(lut_size):
                    val = lut_3d[r_idx, g_idx, b_idx]
                    f.write(f"{val[0]/255.0:.6f} {val[1]/255.0:.6f} {val[2]/255.0:.6f}\n")

    print(f"   .cube file saved!")


def apply_to_folder(lut_3d, target_folder, output_folder):
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
        result_array = apply_grade_3d(img_array, lut_3d)

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
    print("PHASE 1: Learning 3D LUT (Print Correction mode)")
    print("=" * 50)
    
    lut_3d = load_trained_lut()
    if lut_3d is None:
        lut_3d = extract_3d_lut_from_pairs(pairs)
        if lut_3d is not None:
            save_trained_lut(lut_3d)

    if lut_3d is not None:
        # 2. Export .cube file
        if not os.path.exists("Print_Correction.cube"):
            print("\n" + "=" * 50)
            print("PHASE 2: Exporting .cube LUT")
            print("=" * 50)
            save_cube_file(lut_3d, "Print_Correction.cube")
        else:
            print("\n   Print_Correction.cube already exists, skipping export.")

        # 3. Batch-apply to raw photos
        print("\n" + "=" * 50)
        print("PHASE 3: Batch Processing")
        print("=" * 50)
        apply_to_folder(lut_3d, "./Raw_Photos", "./Finished_Photos")

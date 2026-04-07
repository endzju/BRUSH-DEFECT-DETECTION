import numpy as np
import cv2
from pathlib import Path


DEFAULT_PARAMS = {
    'diff_threshold': 40,
    'hsv_s_min': 10,
    'hsv_v_min': 10,
    'morph_open_k': 5,
    'morph_open_iter': 2,
    'morph_close_k': 3,
    'morph_close_iter': 2,
    'min_area': 500
}
VIZ_DIM = (200, 200)

_PATTERNS_CACHE = {}

def calculate_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) for binary masks.
    """
    mask_pred = (mask_pred > 0)
    mask_gt = (mask_gt > 0)

    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def _get_patterns_dir():
    """Returns the absolute path to the pattern directory."""
    return Path(__file__).parent.resolve() / "pattern"


def _load_pattern_cached(color: str, type_name: str = "mean"):
    """Loads and caches the reference pattern from disk."""
    key = f"{color}_{type_name}"
    if key not in _PATTERNS_CACHE:
        p_dir = _get_patterns_dir() / type_name
        p_path = p_dir / f"{color}.png"
        img = cv2.imread(str(p_path))
        if img is None:
            return np.zeros((400, 400, 3), np.uint8)
        _PATTERNS_CACHE[key] = img
    return _PATTERNS_CACHE[key]


def _identify_brush_color(img_bgr: np.ndarray) -> str:
    """
    Identifies brush color by analyzing the Hue of the bristles in HSV space.
    Filters out the dark blue background by requiring high Saturation and Value.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Crop to central ROI where bristles are located (e.g., 100x100 center)
    h, w = hsv.shape[:2]
    roi = hsv[h//2-50:h//2+50, w//2-50:w//2+50]
    
    # Filter for colorful pixels (S > 30 and V > 30) to ignore dark blue background
    bristle_pixels = roi[(roi[:, :, 1] > 30) & (roi[:, :, 2] > 30)]
    
    if len(bristle_pixels) < 10:
        # Fallback if no colorful pixels found: use standard image-wide SAD
        colors = ["blue", "red", "yellow"]
        sums = []
        for color in colors:
            pattern = _load_pattern_cached(color, "mean")
            diff = np.abs(pattern.astype(np.int16) - img_bgr.astype(np.int16)).astype(np.uint8)
            sums.append(np.sum(diff))
        return colors[np.argmin(sums)]

    # Calculate median Hue of the detected bristle pixels
    median_hue = np.median(bristle_pixels[:, 0])
    
    # Map Hue to labels (OpenCV H range is 0-179)
    # Blue: ~108, Yellow: ~18, Red: ~3 or ~178
    if 80 <= median_hue <= 140:
        color = "blue"
    elif 11 <= median_hue <= 40:
        color = "yellow"
    else:
        color = "red"
    
    # print(f"Identified color: {color} (Median Hue: {median_hue:.1f}, Pixels: {len(bristle_pixels)})")
    return color


def _build_diff_mask(bgr_img: np.ndarray, pattern: np.ndarray, 
                     diff_threshold: int, hsv_s_min: int, hsv_v_min: int) -> np.ndarray:
    """Computes difference mask with HSV noise suppression."""
    # Absolute difference in int16
    diff = np.abs(bgr_img.astype(np.int16) - pattern.astype(np.int16))
    
    # Max-channel difference (more sensitive to single-channel shifts)
    diff_max = diff.max(axis=2).astype(np.uint8)
    raw_mask = (diff_max > diff_threshold).astype(np.uint8) * 255

    # HSV guard to ignore dark / grey background areas
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    s_mask = (hsv[:, :, 1] >= hsv_s_min).astype(np.uint8) * 255
    v_mask = (hsv[:, :, 2] >= hsv_v_min).astype(np.uint8) * 255
    guard  = cv2.bitwise_and(s_mask, v_mask)

    return cv2.bitwise_and(raw_mask, guard)


def _morphology_pipeline(mask: np.ndarray, open_k: int, close_k: int, 
                         open_iter: int = 1, close_iter: int = 1) -> np.ndarray:
    """Cleans up the mask using open (noise) and close (holes) operations."""
    k_open  = np.ones((open_k,  open_k),  np.uint8)
    k_close = np.ones((close_k, close_k), np.uint8)

    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=close_iter)
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=open_iter)
    return mask


def _filter_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Removes small noise components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    return filtered


def _visualise(bgr_img: np.ndarray,
               pattern: np.ndarray,
               diff_gray: np.ndarray,
               final_mask: np.ndarray,
               groundtruth_img: np.ndarray | None) -> None:
    """
    Display a side-by-side debug window.
    Shows: Original | Pattern | Max-Diff | Predicted Mask | Ground-Truth Mask.

    Separated from predict() so that the core detection logic can be tested
    or called in batch mode without any GUI dependency.
    """
    
    def _resize(arr):
        return cv2.resize(arr, VIZ_DIM, interpolation=cv2.INTER_AREA)

    gray_3ch = cv2.cvtColor(diff_gray,  cv2.COLOR_GRAY2BGR)
    mask_3ch = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    
    if groundtruth_img is not None:
        gt_3ch = cv2.cvtColor(groundtruth_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        gt_3ch = np.zeros_like(bgr_img)

    row = np.hstack([
        _resize(bgr_img),
        _resize(pattern),
        _resize(gray_3ch),
        _resize(mask_3ch),
        _resize(gt_3ch),
    ])
    cv2.imshow("Result (Original | Pattern | Max-Diff | Mask | GroundTruth)", row)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict(image: np.ndarray, *args, params: dict = None, visualize: bool = False) -> np.ndarray:
    """Thresholding defect segmentation model.
    
    Args:
        image: numpy array of shape (H, W, 3), uint8 RGB image.
        *args: optional ground-truth mask (H, W) uint8.
        params: dict of parameters to override DEFAULT_PARAMS.
        visualize: whether to show the debug window.
        
    Returns:
        Binary defect mask, shape (H, W), dtype uint8, values in {0, 255}.
    """
    p = DEFAULT_PARAMS.copy()
    if params:
        p.update(params)

    groundtruth_img: np.ndarray | None = args[0] if len(args) > 0 else None
    
    bgr = image
    color = _identify_brush_color(bgr)
    pattern = _load_pattern_cached(color, "mean")
    
    raw_mask = _build_diff_mask(bgr, pattern, p['diff_threshold'], p['hsv_s_min'], p['hsv_v_min'])
    
    clean_mask = _morphology_pipeline(raw_mask, 
                                     p['morph_open_k'], p['morph_close_k'],
                                     p.get('morph_open_iter', 1), p.get('morph_close_iter', 1))

    final_mask = _filter_components(clean_mask, p['min_area'])

    if visualize:
        diff_for_viz = np.abs(bgr.astype(np.int16) - pattern.astype(np.int16)) \
                         .max(axis=2).astype(np.uint8)
        _visualise(bgr, pattern, diff_for_viz, final_mask, groundtruth_img)

    return final_mask


def run_grid_search(images: list[np.ndarray], ground_truths: list[np.ndarray], subset_step: int = 10):
    """
    Automatically perform an optimized grid search to find the best parameters.
    Uses heavy subsetting for high speed.
    """
    import itertools
    import time

    param_grid = {
        'diff_threshold': [40, 50, 60],
        'hsv_s_min': [10],
        'hsv_v_min': [10],
        'morph_open_k': [3, 5, 7, 9],
        'morph_close_k': [3, 5, 7, 9],
        'morph_open_iter': [1, 2],
        'morph_close_iter': [1, 2],
        'min_area': [450, 500, 550, 600, 650]
    }

    preped_data = []
    for img, gt in zip(images[::subset_step], ground_truths[::subset_step]):
        color = _identify_brush_color(img)
        pattern = _load_pattern_cached(color, "mean")
        preped_data.append((img, pattern, gt))

    keys = param_grid.keys()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    
    total_trials = len(combinations)
    print(f"Starting Optimized Grid Search: {total_trials} combinations on {len(preped_data)} images.")

    best_iou = -1.0
    best_params = None
    start_time = time.time()

    for i, p in enumerate(combinations):
        current_ious = []
        for bgr, pattern, gt in preped_data:
            raw_mask = _build_diff_mask(bgr, pattern, p['diff_threshold'], p['hsv_s_min'], p['hsv_v_min'])
            clean_mask = _morphology_pipeline(raw_mask, 
                                             p['morph_open_k'], p['morph_close_k'],
                                             p['morph_open_iter'], p['morph_close_iter'])
            final_mask = _filter_components(clean_mask, p['min_area'])
            current_ious.append(calculate_iou(final_mask, gt))
        
        avg_iou = np.mean(current_ious)
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_params = p
            print(f"[Trial {i+1}/{total_trials}] NEW BEST: IoU={best_iou:.4f} | Params: {best_params}")
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + 1)) * (total_trials - (i + 1))
            print(f"[Progress {i+1}/{total_trials}] Current IoU={avg_iou:.4f} | Best IoU={best_iou:.4f} | ETA: {remaining/60:.2f} min")

    print("\n" + "="*40)
    print("GRID SEARCH COMPLETE")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*40)

    return best_params

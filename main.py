from model import *

# --- SETTINGS ---
RUN_GRID_SEARCH = False
SUBSET_STEP = 6
# ----------------

def save_result_grid(img_idx: int, bgr_img: np.ndarray, diff_gray: np.ndarray,
                     mask: np.ndarray, gt: np.ndarray, output_dir: Path):
    """Saves a 2x2 grid: Original | Max-Diff // Mask | Ground-Truth."""
    def _resize(arr):
        return cv2.resize(arr, VIZ_DIM, interpolation=cv2.INTER_AREA)

    orig  = _resize(bgr_img)
    diff3 = _resize(cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2BGR))
    mask3 = _resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    gt3   = _resize(cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_GRAY2BGR))

    top    = np.hstack([orig,  diff3])
    bottom = np.hstack([mask3, gt3])
    grid   = np.vstack([top, bottom])

    out_path = output_dir / f"{img_idx:03d}_result.png"
    cv2.imwrite(str(out_path), grid)


def main():
    script_dir = Path(__file__).parent.resolve()
    defective_path = script_dir / "toothbrush" / "train" / "defective"
    ground_truth_path = script_dir / "toothbrush" / "ground_truth" / "defective"
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    if RUN_GRID_SEARCH:
        print(f"--- STARTING GRID SEARCH (Step: {SUBSET_STEP}) ---")
        all_images, all_gts = [], []
        for i in range(30):
            img = cv2.imread(str(defective_path / f"{i:03d}.png"))
            gt  = cv2.imread(str(ground_truth_path / f"{i:03d}_mask.png"), 0)
            if img is not None and gt is not None:
                all_images.append(img)
                all_gts.append(gt)

        if not all_images:
            print("Error: No images found!")
            return

        best_p = run_grid_search(all_images, all_gts, subset_step=SUBSET_STEP)
        print(f"\nRecommended parameters: {best_p}")
        return

    print("--- RUNNING EVALUATION WITH CURRENT PARAMETERS ---")
    total_iou = 0.0
    count = 0

    for i in range(30):
        img = cv2.imread(str(defective_path / f"{i:03d}.png"))
        gt  = cv2.imread(str(ground_truth_path / f"{i:03d}_mask.png"), 0)

        if img is not None and gt is not None:
            mask = predict(img, gt, visualize=False)
            iou  = calculate_iou(mask, gt)
            print(f"Image {i:03d}: IoU = {iou:.4f}")

            color = identify_brush_color(img)
            pattern = load_pattern_cached(color, "mean")
            diff = np.abs(img.astype(np.int16) - pattern.astype(np.int16)) \
                        .max(axis=2).astype(np.uint8)

            save_result_grid(i, img, diff, mask, gt, results_dir)

            total_iou += iou
            count += 1

    if count > 0:
        print(f"\nFinal Mean IoU: {total_iou/count:.4f}")
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
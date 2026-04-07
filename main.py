from model import *

# --- SETTINGS ---
RUN_GRID_SEARCH = False
SUBSET_STEP = 6
# ----------------

def main():
    script_dir = Path(__file__).parent.resolve()
    defective_path = script_dir / "toothbrush" / "train" / "defective"
    ground_truth_path = script_dir / "toothbrush" / "ground_truth" / "defective"
    
    if RUN_GRID_SEARCH:
        print(f"--- STARTING GRID SEARCH (Step: {SUBSET_STEP}) ---")
        all_images = []
        all_gts = []
        for i in range(30):
            img = cv2.imread(str(defective_path / f"{i:03d}.png"))
            gt = cv2.imread(str(ground_truth_path / f"{i:03d}_mask.png"), 0)
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
        gt = cv2.imread(str(ground_truth_path / f"{i:03d}_mask.png"), 0)
        
        if img is not None and gt is not None:
            mask = predict(img, gt, visualize=True)
            iou = calculate_iou(mask, gt)
            print(f"Image {i:03d}: IoU = {iou:.4f}")
            total_iou += iou
            count += 1

    if count > 0:
        print(f"\nFinal Mean IoU: {total_iou/count:.4f}")

if __name__ == "__main__":
    main()

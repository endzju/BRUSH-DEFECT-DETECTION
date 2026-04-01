from utils import *
from model import *
def main():

    script_dir = Path(__file__).parent.resolve()
    good_path = script_dir / "toothbrush" / "train" / "good"
    defective_path = script_dir / "toothbrush" / "train" / "defective"
    ground_truth_path = script_dir / "toothbrush" / "ground_truth" / "defective"
    
    images = []
    ground_truths = []
    for i in range(30):
        img = cv2.imread(f"{defective_path}/{i:03d}.png")
        gt = cv2.imread(f"{ground_truth_path}/{i:03d}_mask.png", 0)
        images.append(img)
        ground_truths.append(gt)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predict(img_rgb, gt) # Pass rgb image, gt is optional, only for visualization

if __name__ == "__main__":
    main()

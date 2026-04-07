from utils import *
from model import *
def main():

    script_dir = Path(__file__).parent.resolve()
    good_path = script_dir / "toothbrush" / "train" / "good"
    defective_path = script_dir / "toothbrush" / "train" / "defective"
    ground_truth_path = script_dir / "toothbrush" / "ground_truth" / "defective"

    init_colors()
    
    images = []
    ground_truths = []
    for i in range(30):
        img = cv2.imread(f"{defective_path}/{i:03d}.png")
        gt = cv2.imread(f"{ground_truth_path}/{i:03d}_mask.png", 0)
        
        images.append(img)
        ground_truths.append(gt)

        image, gray_3ch, mask_3ch = predict(img)
        gt_3ch = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        dim = (300, 300)

        img_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        gray_resized = cv2.resize(gray_3ch, dim, interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_3ch, dim, interpolation=cv2.INTER_AREA)
        gt_resized = cv2.resize(gt_3ch, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow('Original', img_resized)
        cv2.imshow('Gray diff', gray_resized)
        cv2.imshow('Mask', mask_resized)
        cv2.imshow('Ground Truth', gt_resized)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 27: # Esc
            break

if __name__ == "__main__":
    main()

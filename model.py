import numpy as np
import cv2
from utils import *
import matplotlib as plt


def predict(image, groundtruth_img=None):
    """Simple thresholding segmentation model.

    Args:
        image: numpy array of shape (H, W, 3), uint8 RGB image.

    Returns:
        Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.
    """
    image = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    pattern_type = "median"
    pattern_gauss = False
    pattern_color = get_color(image, "mean")

    pattern = get_pattern(mean_or_median=pattern_type, color=pattern_color, gauss=pattern_gauss)

    diff = abs(image.astype(np.int16) - pattern.astype(np.uint16)).astype(np.uint8)
    gray = np.mean(diff, axis=2).astype(np.uint8)
    mask = (gray > 30).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_size = 200

    filtered_mask = np.zeros(mask.shape, np.uint8)

    for i in range(1, num_labels): # zaczynamy od 1, bo 0 to tło
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 255

    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask_3ch = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)
    groundtruth_img = groundtruth_img.astype(np.uint8)
    gt_3ch = cv2.cvtColor(groundtruth_img, cv2.COLOR_GRAY2BGR)

    dim = (500, 500)
    window_content = np.hstack((cv2.resize(image, dim, interpolation=cv2.INTER_AREA),
                                cv2.resize(gray_3ch, dim, interpolation=cv2.INTER_AREA),
                                cv2.resize(mask_3ch, dim, interpolation=cv2.INTER_AREA),
                                cv2.resize(gt_3ch, dim, interpolation=cv2.INTER_AREA)
                                ))
    cv2.imshow('result', window_content)
    cv2.resizeWindow('result', dim[0]*4, dim[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return mask

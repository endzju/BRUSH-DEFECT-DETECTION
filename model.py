import numpy as np
import cv2
from utils import *
import matplotlib as plt


def predict(image):
    """Simple thresholding segmentation model.

    Args:
        image: numpy array of shape (H, W, 3), uint8 RGB image.
    """

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

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 255

    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask_3ch = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)

    return image, gray_3ch, mask_3ch

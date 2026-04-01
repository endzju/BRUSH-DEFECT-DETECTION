import cv2
import numpy as np
from pathlib import Path
import os
from utils import *

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    good_path = script_dir / "toothbrush" / "train" / "good"
    defective_path = script_dir / "toothbrush" / "train" / "defective"
    images = []
    for i in range(60):
        img = cv2.imread(f"{good_path}/{i:03d}.png")
        images.append(img)
        
    create_mean_median_mask(images[0:20], "blue")
    create_mean_median_mask(images[20:40], "red")
    create_mean_median_mask(images[40:60], "yellow")
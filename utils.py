import cv2
import numpy as np
from pathlib import Path
import os

def create_mean_median_mask(images: list[np.ndarray], color: str):
    script_dir = Path(__file__).parent.resolve()
    mean_dir = script_dir / "pattern" / "mean"
    median_dir = script_dir / "pattern" / "median"

    mean_dir.mkdir(parents=True, exist_ok=True)
    median_dir.mkdir(parents=True, exist_ok=True)

    mean_output = np.zeros(images[0].shape, np.uint16)
    median_output = np.zeros(images[0].shape, np.uint16)

    for img in images:
        mean_output += img
    mean_output = (mean_output // len(images)).astype(np.uint8)

    stack = np.stack(images, axis=0)
    median_output = np.median(stack, axis=0).astype(np.uint8)

    mean_file_path = mean_dir / (color + ".png")
    median_file_path = median_dir / (color + ".png")
    mean_gauss_file_path = mean_dir / (color + "_gauss.png")
    median_gauss_file_path = median_dir / (color + "_gauss.png")

    mean_gauss_output = cv2.GaussianBlur(mean_output, (9, 9), sigmaX=0)
    median_gauss_output = cv2.GaussianBlur(median_output, (9, 9), sigmaX=0)

    cv2.imwrite(str(mean_file_path), mean_output)
    cv2.imwrite(str(median_file_path), median_output)
    cv2.imwrite(str(mean_gauss_file_path), mean_gauss_output)
    cv2.imwrite(str(median_gauss_file_path), median_gauss_output)

def get_color(img: np.ndarray, median_or_mean: str) -> str:
    colors = ["blue", "red", "yellow"]
    means = []
    medians = []
    script_dir = Path(__file__).parent.resolve()
    mean_dir = script_dir / "pattern" / "mean"
    median_dir = script_dir / "pattern" / "median"
    for color in colors:
        img_mean_path = mean_dir / (color+".png")
        img_median_path = median_dir / (color+".png")
        means.append(cv2.imread(img_mean_path))
        medians.append(cv2.imread(img_median_path))
    if median_or_mean == "mean":
        patterns = means
    elif median_or_mean == "median":
        patterns = medians
    else:
        raise ValueError("median_or_mean must be 'mean' or 'median'")
    sums = []
    for pattern_img in means:
        diff = abs(pattern_img.astype(np.int16) - img.astype(np.int16)).astype(np.uint8)
        sums.append(np.sum(diff))
    color_idx = np.argmin(sums)
    return colors[color_idx]

def get_pattern(mean_or_median: str, color: str, gauss:bool=False) -> np.ndarray:
    if mean_or_median not in ['median', 'mean']:
        raise ValueError("median_or_mean must be 'mean' or 'median'")
    if color not in ["blue", "red", "yellow"]:
        raise ValueError("color must be 'blue', 'red' or 'yellow'")
    script_dir = Path(__file__).parent.resolve()
    dir = script_dir / "pattern" / mean_or_median
    name = color
    if gauss:
        name += "_gauss"
    name += ".png"
    file_name = str(dir / name)
    img_pattern = cv2.imread(file_name)
    return img_pattern
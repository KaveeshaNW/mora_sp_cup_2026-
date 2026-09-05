import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pywt

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NOISE_SUFFIX = "_noise"


def repair_defect_pixels_quality(img_bgr: np.ndarray, base_multiplier: float = 3.5) -> np.ndarray:
    """
    Maximizes SSIM/PSNR by using adaptive local median absolute deviation (MAD)
    and Scharr edge-aware masking to protect fine geometric textures.
    """
    # Calculate median on the original BGR image
    median_bgr = cv2.medianBlur(img_bgr, 3)
    
    # Convert to grayscale float32 for accurate gradient and deviation math
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    median_gray = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 1. Adaptive Local Variance (MAD)
    abs_deviation = np.abs(gray - median_gray)
    local_mad = cv2.GaussianBlur(abs_deviation, (3, 3), 0)
    local_mad = np.clip(local_mad, 1.0, None) 
    
    # 2. Structural Edge Mapping (Scharr)
    scharr_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    edge_magnitude = cv2.magnitude(scharr_x, scharr_y)
    
    # 3. Edge Dilation (Boundary Protection)
    edge_map = np.clip(edge_magnitude / 255.0, 0, 1)
    edge_map = cv2.dilate(edge_map, np.ones((3, 3), np.float32))
    
    # 4. Dynamic Threshold Calculation
    dynamic_threshold = base_multiplier * local_mad * (1.0 + edge_map * 6.0)
    
    # 5. Coupled BGR Replacement
    outlier_mask = abs_deviation > dynamic_threshold
    return np.where(outlier_mask[..., None], median_bgr, img_bgr)


def wavelet_bayesshrink_denoise_channel(channel: np.ndarray, wavelet: str = "db4", levels: int = 2) -> np.ndarray:
    """Applies multi-level 2D DWT BayesShrink adaptive soft-thresholding to a single channel."""
    channel_float = channel.astype(np.float32) / 255.0
    coeffs = pywt.wavedec2(channel_float, wavelet=wavelet, level=levels)
    
    # Estimate noise variance sigma using Robust MAD on the finest detail subband (HH1)
    hh1 = coeffs[-1][2]
    sigma = np.median(np.abs(hh1)) / 0.6745 + 1e-8
    
    new_coeffs = [coeffs[0]]
    for detail_level in coeffs[1:]:
        denoised_details = []
        for subband in detail_level:
            var_y = np.var(subband)
            var_x = max(0.0, var_y - sigma**2)
            if var_x == 0:
                t = np.max(np.abs(subband))
            else:
                t = (sigma**2) / np.sqrt(var_x)
            
            subband_thresh = pywt.threshold(subband, value=t, mode="soft")
            denoised_details.append(subband_thresh)
        new_coeffs.append(tuple(denoised_details))
    
    reconstructed = pywt.waverec2(new_coeffs, wavelet=wavelet)
    reconstructed = np.clip(reconstructed, 0.0, 1.0) * 255.0
    return reconstructed.astype(np.uint8)


def process_single_image(img_bgr: np.ndarray) -> np.ndarray:
    """Executes the complete classical DSP pipeline on a single image."""
    # Step 1: Clean hot/dead pixels with the high SSIM/PSNR function
    clean_bgr = repair_defect_pixels_quality(img_bgr, base_multiplier=3.5)
    
    # Step 2: Convert to Lab color space for perceptual luminance-chroma separation
    lab = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Step 3: Luminance detail preservation (Wavelet BayesShrink + Mild Bilateral)
    l_wavelet = wavelet_bayesshrink_denoise_channel(l_channel, wavelet="db4", levels=2)
    l_denoised = cv2.bilateralFilter(l_wavelet, d=5, sigmaColor=25, sigmaSpace=25)
    
    # Step 4: Chrominance noise suppression (Aggressive Bilateral Filter on a & b)
    a_denoised = cv2.bilateralFilter(a_channel, d=9, sigmaColor=60, sigmaSpace=60)
    b_denoised = cv2.bilateralFilter(b_channel, d=9, sigmaColor=60, sigmaSpace=60)
    
    # Step 5: Merge and convert back to BGR
    merged_lab = cv2.merge([l_denoised, a_denoised, b_denoised])
    bgr_reconstructed = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2BGR)
    
    # Step 6: Fast Non-Local Means refinement with compact search window for high SSIM
    final_output = cv2.fastNlMeansDenoisingColored(
        bgr_reconstructed,
        None,
        h=6.0,
        hColor=10.0,
        templateWindowSize=5,
        searchWindowSize=13
    )
    
    return final_output


def strip_noise_suffix(stem: str) -> str:
    """Strips the '_noise' suffix to match the required submission output format."""
    if stem.lower().endswith(NOISE_SUFFIX):
        return stem[:-len(NOISE_SUFFIX)]
    return stem


def worker_task(task_data: Tuple[Path, Path]) -> None:
    src_path, dst_path = task_data
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    denoised_img = process_single_image(img)
    cv2.imwrite(str(dst_path), denoised_img)


def main():
    parser = argparse.ArgumentParser(description="Mora SP Cup 2026 Classical Denoising Pipeline")
    parser.add_argument(
        "--noise_dir",
        "--input_dir",
        dest="noise_dir",
        required=True,
        type=Path,
        help="Directory containing input noisy images",
    )
    parser.add_argument(
        "--denoised_dir",
        "--output_dir",
        dest="denoised_dir",
        required=True,
        type=Path,
        help="Directory where denoised outputs will be saved",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="Number of parallel CPU worker processes",
    )
    args = parser.parse_args()

    args.denoised_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([
        p for p in args.noise_dir.glob("*") if p.suffix.lower() in VALID_EXTENSIONS
    ])

    if not image_paths:
        print(f"Error: No valid images found in {args.noise_dir}")
        sys.exit(1)

    tasks = [
        (p, args.denoised_dir / f"{strip_noise_suffix(p.stem)}.png")
        for p in image_paths
    ]

    # Prevent OpenCV internal thread contention when multiprocessing
    cv2.setNumThreads(1)

    print(f"Processing {len(tasks)} images using {args.num_workers} CPU workers...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        list(pool.map(worker_task, tasks))

    total_time = time.time() - t0
    print(f"Completed in {total_time:.2f}s ({total_time / len(tasks) * 1000:.1f} ms/image)")
    print(f"Outputs saved to: {args.denoised_dir}")


if __name__ == "__main__":
    main()
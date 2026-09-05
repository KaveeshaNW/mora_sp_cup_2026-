import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import onnxruntime as ort
import cv2
import numpy as np

# Dynamically locate the model file inside the scripts/ folder
script_dir = Path(__file__).parent
model_path = str(script_dir / "nafnet_denoise.onnx")

# Initialize ONNX Session Globally
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1 
ort_session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
input_name = ort_session.get_inputs()[0].name

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NOISE_SUFFIX = "_noise"


def repair_defect_pixels_quality(img_bgr: np.ndarray, base_multiplier: float = 3.5) -> np.ndarray:
    """
    Maximizes SSIM/PSNR by using adaptive local median absolute deviation (MAD)
    and Scharr edge-aware masking to protect fine geometric textures.
    """
    median_bgr = cv2.medianBlur(img_bgr, 3)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    median_gray = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    abs_deviation = np.abs(gray - median_gray)
    local_mad = cv2.GaussianBlur(abs_deviation, (3, 3), 0)
    local_mad = np.clip(local_mad, 1.0, None) 
    
    scharr_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    edge_magnitude = cv2.magnitude(scharr_x, scharr_y)
    
    edge_map = np.clip(edge_magnitude / 255.0, 0, 1)
    edge_map = cv2.dilate(edge_map, np.ones((3, 3), np.float32))
    
    dynamic_threshold = base_multiplier * local_mad * (1.0 + edge_map * 6.0)
    
    outlier_mask = abs_deviation > dynamic_threshold
    return np.where(outlier_mask[..., None], median_bgr, img_bgr)


class ImageTiler:
    """Handles splitting high-res images into overlapping patches and seamless blending."""
    def __init__(self, tile_size: int = 256, overlap: int = 32):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.window = self._create_2d_window(tile_size)

    def _create_2d_window(self, size: int) -> np.ndarray:
        h = np.hanning(size)
        window_2d = np.outer(h, h)
        return np.clip(window_2d, 1e-4, 1.0)[:, :, None].astype(np.float32)

    def split(self, img_bgr: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]:
        h, w, c = img_bgr.shape
        
        pad_h = (self.stride - ((h - self.tile_size) % self.stride)) % self.stride if h > self.tile_size else (self.tile_size - h)
        pad_w = (self.stride - ((w - self.tile_size) % self.stride)) % self.stride if w > self.tile_size else (self.tile_size - w)

        padded = np.pad(img_bgr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        pad_h_total, pad_w_total, _ = padded.shape

        tiles = []
        coords = []

        for y in range(0, pad_h_total - self.tile_size + 1, self.stride):
            for x in range(0, pad_w_total - self.tile_size + 1, self.stride):
                patch = padded[y : y + self.tile_size, x : x + self.tile_size]
                tiles.append(patch)
                coords.append((y, x))

        return tiles, coords, (pad_h_total, pad_w_total)

    def merge(self, tiles: list[np.ndarray], coords: list[tuple[int, int]], 
              padded_shape: tuple[int, int], orig_shape: tuple[int, int]) -> np.ndarray:
        pad_h, pad_w = padded_shape
        c = tiles[0].shape[2]
        
        canvas = np.zeros((pad_h, pad_w, c), dtype=np.float32)
        weight_map = np.zeros((pad_h, pad_w, 1), dtype=np.float32)

        for tile, (y, x) in zip(tiles, coords):
            tile_float = tile.astype(np.float32)
            canvas[y : y + self.tile_size, x : x + self.tile_size] += tile_float * self.window
            weight_map[y : y + self.tile_size, x : x + self.tile_size] += self.window

        canvas /= np.clip(weight_map, 1e-6, None)
        orig_h, orig_w = orig_shape
        output = canvas[:orig_h, :orig_w]
        return np.clip(output, 0, 255).astype(np.uint8)


def process_single_image(img_bgr: np.ndarray) -> np.ndarray:
    """Executes the tiling and neural network inference pipeline."""
    
    # Stage 1: Initialize Tiler and split the RAW image directly
    tiler = ImageTiler(tile_size=256, overlap=32)
    tiles, coords, padded_shape = tiler.split(img_bgr)
    
    # Stage 2: Deep Learning Inference (Neural)
    processed_tiles = []
    for tile in tiles:
        # 1. Convert incoming BGR to RGB for the network
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        
        # 2. Format for CNN: HWC uint8 -> NCHW float32
        tile_float = tile_rgb.astype(np.float32) / 255.0
        tile_nchw = np.transpose(tile_float, (2, 0, 1))[np.newaxis, ...]
        
        # 3. Execute ONNX graph
        ort_outs = ort_session.run(None, {input_name: tile_nchw})
        
        # 4. Format for Merger: NCHW float32 -> HWC uint8
        denoised_nchw = ort_outs[0][0]
        denoised_hwc = np.transpose(denoised_nchw, (1, 2, 0))
        denoised_uint8 = np.clip(denoised_hwc * 255.0, 0, 255).astype(np.uint8)
        
        # 5. Convert the clean RGB tensor back to BGR for OpenCV
        denoised_bgr = cv2.cvtColor(denoised_uint8, cv2.COLOR_RGB2BGR)
        
        processed_tiles.append(denoised_bgr)
    
    # Stage 3: Reassemble tiles seamlessly
    reconstructed_bgr = tiler.merge(
        processed_tiles, 
        coords, 
        padded_shape, 
        (img_bgr.shape[0], img_bgr.shape[1])
    )
    
    return reconstructed_bgr


def strip_noise_suffix(stem: str) -> str:
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
    parser = argparse.ArgumentParser(description="Hybrid Pipeline: Pre-Processing & Tiling")
    parser.add_argument("--noise_dir", "--input_dir", dest="noise_dir", required=True, type=Path)
    parser.add_argument("--denoised_dir", "--output_dir", dest="denoised_dir", required=True, type=Path)
    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() - 1))
    args = parser.parse_args()

    args.denoised_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in args.noise_dir.glob("*") if p.suffix.lower() in VALID_EXTENSIONS])

    if not image_paths:
        print(f"Error: No valid images found in {args.noise_dir}")
        sys.exit(1)

    tasks = [(p, args.denoised_dir / f"{strip_noise_suffix(p.stem)}.png") for p in image_paths]

    # Prevent OpenCV internal thread contention when multiprocessing
    cv2.setNumThreads(1)

    print(f"Processing {len(tasks)} images using {args.num_workers} CPU workers...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        list(pool.map(worker_task, tasks))

    total_time = time.time() - t0
    print(f"Completed in {total_time:.2f}s ({total_time / len(tasks) * 1000:.1f} ms/image)")


if __name__ == "__main__":
    main()
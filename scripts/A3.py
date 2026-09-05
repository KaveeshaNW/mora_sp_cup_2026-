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


class ImageTiler:
    """Handles splitting high-res images into overlapping patches and seamless blending."""
    def __init__(self, tile_size: int = 256, overlap: int = 128):
        # OPTIMIZATION: Overlap increased to 128 (50%) for mathematically perfect Hanning summation
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


def infer_with_tta(tile_rgb: np.ndarray) -> np.ndarray:
    """Applies x4 Test-Time Augmentation to maximize PSNR/SSIM by averaging predictions."""
    tile_float = tile_rgb.astype(np.float32) / 255.0
    
    # 1. Define x4 geometric augmentations
    aug_inputs = [
        tile_float,                                   # Original
        np.fliplr(tile_float),                        # Horizontal Flip
        np.flipud(tile_float),                        # Vertical Flip
        np.flipud(np.fliplr(tile_float))              # 180 Rotation
    ]
    
    aug_results = []
    for v in aug_inputs:
        v_nchw = np.transpose(v.copy(), (2, 0, 1))[np.newaxis, ...]
        ort_outs = ort_session.run(None, {input_name: v_nchw})
        out_hwc = np.transpose(ort_outs[0][0], (1, 2, 0))
        aug_results.append(out_hwc)
        
    # 2. Inverse the augmentations to align predictions back to original geometry
    r_orig = aug_results[0]
    r_hflip = np.fliplr(aug_results[1])
    r_vflip = np.flipud(aug_results[2])
    r_180 = np.fliplr(np.flipud(aug_results[3]))
    
    # 3. Average the predictions (Reduces hallucinated noise and color shifts)
    ensemble = (r_orig + r_hflip + r_vflip + r_180) / 4.0
    return np.clip(ensemble * 255.0, 0, 255).astype(np.uint8)


def process_single_image(img_bgr: np.ndarray) -> np.ndarray:
    """Executes the optimized tiling and x4 TTA neural network pipeline."""
    
    tiler = ImageTiler(tile_size=256, overlap=128)
    tiles, coords, padded_shape = tiler.split(img_bgr)
    
    processed_tiles = []
    for tile in tiles:
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        
        # OPTIMIZATION: Execute Test-Time Augmentation instead of single pass
        denoised_rgb_uint8 = infer_with_tta(tile_rgb)
        
        denoised_bgr = cv2.cvtColor(denoised_rgb_uint8, cv2.COLOR_RGB2BGR)
        processed_tiles.append(denoised_bgr)
    
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
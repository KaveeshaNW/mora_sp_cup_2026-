"""
scripts/denoise.py
Mora SP Cup 2026 - High-Efficiency Hybrid CNN Pipeline
Targets 0.60+ Composite Score under CPU latency constraints.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort

# Model setup
script_dir = Path(__file__).parent
model_path = str(script_dir / "nafnet_denoise.onnx")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NOISE_SUFFIX = "_noise"


def get_ort_session():
    """Configures an optimized single-process CPU ONNX session."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    return session


def clean_hot_pixels_fast(img_bgr: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Vectorized impulse noise suppression.
    Crucial for PSNR: Eliminates hot sensor pixels that CNNs fail to invert.
    """
    med = cv2.medianBlur(img_bgr, 3)
    diff = cv2.absdiff(img_bgr, med)
    mask = np.any(diff > threshold, axis=2, keepdims=True)
    return np.where(mask, med, img_bgr)


class BatchedImageTiler:
    """Splits 992x992 images into 384x384 patches with smooth Hanning reconstruction."""

    def __init__(self, tile_size: int = 384, overlap: int = 48):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.window = self._build_hanning_window(tile_size)

    def _build_hanning_window(self, size: int) -> np.ndarray:
        h = np.hanning(size)
        w2d = np.outer(h, h)
        return np.clip(w2d, 1e-3, 1.0)[:, :, None].astype(np.float32)

    def extract_batch(
        self, img_rgb_float: np.ndarray
    ) -> Tuple[np.ndarray, list, Tuple[int, int]]:
        h, w, _ = img_rgb_float.shape

        pad_h = (self.stride - ((h - self.tile_size) % self.stride)) % self.stride
        pad_w = (self.stride - ((w - self.tile_size) % self.stride)) % self.stride

        padded = np.pad(
            img_rgb_float, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
        )
        pad_h_total, pad_w_total, _ = padded.shape

        tiles = []
        coords = []
        for y in range(0, pad_h_total - self.tile_size + 1, self.stride):
            for x in range(0, pad_w_total - self.tile_size + 1, self.stride):
                patch = padded[y : y + self.tile_size, x : x + self.tile_size]
                # Transpose HWC -> CHW
                tiles.append(np.transpose(patch, (2, 0, 1)))
                coords.append((y, x))

        batch_tensor = np.ascontiguousarray(np.stack(tiles, axis=0), dtype=np.float32)
        return batch_tensor, coords, (pad_h_total, pad_w_total)

    def reconstruct_image(
        self,
        batch_out: np.ndarray,
        coords: list,
        padded_shape: Tuple[int, int],
        orig_shape: Tuple[int, int],
    ) -> np.ndarray:
        pad_h, pad_w = padded_shape
        canvas = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        weight_map = np.zeros((pad_h, pad_w, 1), dtype=np.float32)

        for i, (y, x) in enumerate(coords):
            # Transpose CHW -> HWC
            tile = np.transpose(batch_out[i], (1, 2, 0))
            canvas[y : y + self.tile_size, x : x + self.tile_size] += (
                tile * self.window
            )
            weight_map[y : y + self.tile_size, x : x + self.tile_size] += self.window

        canvas /= np.clip(weight_map, 1e-5, None)
        out = canvas[: orig_shape[0], : orig_shape[1]]
        return np.clip(out, 0.0, 1.0)


def process_image(img_bgr: np.ndarray, session, input_name: str) -> np.ndarray:
    orig_h, orig_w = img_bgr.shape[:2]

    # Step 1: Pre-suppress impulse defect noise
    clean_bgr = clean_hot_pixels_fast(img_bgr, threshold=32)

    # Step 2: Convert to RGB float [0.0, 1.0]
    img_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Step 3: Extract batched tiles (Only 9 tiles for 992x992 image)
    tiler = BatchedImageTiler(tile_size=384, overlap=48)
    batch_tensor, coords, padded_shape = tiler.extract_batch(img_rgb)

    # Step 4: Batched vectorized inference
    ort_outs = session.run(None, {input_name: batch_tensor})
    denoised_batch = ort_outs[0]

    # Step 5: Seamless Hanning blend
    denoised_rgb = tiler.reconstruct_image(
        denoised_batch, coords, padded_shape, (orig_h, orig_w)
    )

    # Step 6: Post-Refinement Chroma Cleanup
    # Eliminates low-frequency color blotches that degrade delta SSIM
    denoised_uint8 = (denoised_rgb * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(denoised_uint8, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab)
    a = cv2.bilateralFilter(a, d=5, sigmaColor=30, sigmaSpace=30)
    b = cv2.bilateralFilter(b, d=5, sigmaColor=30, sigmaSpace=30)
    final_rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2RGB)

    return cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)


def strip_noise_suffix(stem: str) -> str:
    if stem.lower().endswith(NOISE_SUFFIX):
        return stem[: -len(NOISE_SUFFIX)]
    return stem


def worker_entry(task_args: Tuple[Path, Path]) -> None:
    src_path, dst_path = task_args
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        return

    # Initialize per-process ONNX session
    session = get_ort_session()
    input_name = session.get_inputs()[0].name

    result_bgr = process_image(img, session, input_name)
    cv2.imwrite(str(dst_path), result_bgr)


def main():
    parser = argparse.ArgumentParser(description="Mora SP Cup Denoising Pipeline")
    parser.add_argument(
        "--noise_dir", "--input_dir", dest="noise_dir", required=True, type=Path
    )
    parser.add_argument(
        "--denoised_dir",
        "--output_dir",
        dest="denoised_dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--num_workers", type=int, default=max(1, min(4, os.cpu_count() // 2))
    )
    args = parser.parse_args()

    args.denoised_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        [p for p in args.noise_dir.glob("*") if p.suffix.lower() in VALID_EXTENSIONS]
    )

    if not paths:
        print(f"[-] No valid images found in {args.noise_dir}")
        sys.exit(1)

    tasks = [
        (p, args.denoised_dir / f"{strip_noise_suffix(p.stem)}.png") for p in paths
    ]

    cv2.setNumThreads(1)
    print(f"[*] Processing {len(tasks)} images with {args.num_workers} processes...")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        list(pool.map(worker_entry, tasks))

    elapsed = time.time() - t0
    print(
        f"[+] Finished in {elapsed:.2f}s (Avg: {elapsed/len(tasks)*1000:.1f} ms/image)"
    )


if __name__ == "__main__":
    main()

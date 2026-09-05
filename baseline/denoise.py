"""
denoise.py
-----------------------
RUN ON: CPU only. No training, no GPU, no model weights.

Fully classical low-light denoising pipeline:
    defect-pixel correction -> Non-Local Means denoise

No brightness/tone correction is applied -- input and ground truth are
both naturally dim evening/night images, differing mainly by noise, so
there is no exposure gap to fix.

Reads every `<id>_noise.<ext>` file from --input_dir and writes `<id>.<ext>`
to --output_dir.

USAGE:
    python denoise.py --input_dir noise_test --output_dir my_outputs
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VALID_EXT = {".png", ".jpg", ".jpeg"}
NOISE_SUFFIX = "_noise"


def correct_defect_pixels(img_float: np.ndarray, threshold: float = 0.25,
                           kernel_size: int = 3) -> np.ndarray:
    """Generic local-median outlier repair -- no knowledge of any actual
    defect mask, location, or percentage."""
    corrected = img_float.copy()
    for c in range(img_float.shape[2]):
        channel_uint8 = (img_float[..., c] * 255).astype(np.uint8)
        median_uint8 = cv2.medianBlur(channel_uint8, kernel_size)
        median = median_uint8.astype(np.float32) / 255.0
        deviation = np.abs(img_float[..., c] - median)
        outlier_mask = deviation > threshold
        corrected[..., c] = np.where(outlier_mask, median, img_float[..., c])
    return corrected


def denoise_nlm(img_uint8: np.ndarray, h: float = 10.0, h_color: float = 10.0,
                 template_window: int = 7, search_window: int = 21) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img_uint8, None, h, h_color,
                                            template_window, search_window)


def denoise_bilateral(img_uint8: np.ndarray, d: int = 9,
                       sigma_color: float = 50, sigma_space: float = 50) -> np.ndarray:
    return cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)


def process_image(img_float: np.ndarray, args) -> np.ndarray:
    if not args.skip_defect_correction:
        img_float = correct_defect_pixels(img_float, threshold=0.25)

    if args.denoise_method != "none":
        img_uint8 = (img_float * 255).astype(np.uint8)
        if args.denoise_method == "nlm":
            img_uint8 = denoise_nlm(img_uint8, h=args.nlm_h, h_color=args.nlm_h)
        elif args.denoise_method == "bilateral":
            img_uint8 = denoise_bilateral(img_uint8)
        img_float = img_uint8.astype(np.float32) / 255.0

    return img_float


def strip_noise_suffix(stem: str) -> str:
    if stem.lower().endswith(NOISE_SUFFIX):
        return stem[: -len(NOISE_SUFFIX)]
    return stem


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--skip_defect_correction", action="store_true")
    ap.add_argument("--denoise_method", choices=["nlm", "bilateral", "none"], default="nlm")
    ap.add_argument("--nlm_h", type=float, default=10.0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted([p for p in args.input_dir.glob("*") if p.suffix.lower() in VALID_EXT])
    if not paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    total_time = 0.0
    for path in paths:
        img = Image.open(path).convert("RGB")
        img_float = np.asarray(img).astype(np.float32) / 255.0

        start = time.time()
        result = process_image(img_float, args)
        total_time += time.time() - start

        out_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        out_id = strip_noise_suffix(path.stem)
        Image.fromarray(out_uint8).save(args.output_dir / f"{out_id}.png")

    print(f"Processed {len(paths)} images in {total_time:.2f}s "
          f"({total_time/len(paths)*1000:.1f} ms/image, CPU, no training)")
    print(f"Results written to: {args.output_dir}")
    print("Filenames match the required submission convention (<id>.png, no suffix).")


if __name__ == "__main__":
    main()
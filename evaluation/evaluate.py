#!/usr/bin/env python3
"""
evaluate.py
-----------------------
Computes the denoising score for a set of images. This is a self-check
tool: it reads the noisy input, your denoised output, and the ground-truth
images, then reports the mean PSNR, SSIM, and composite score.

USAGE (three required paths):
    python evaluate.py \
        --noisy_dir noise_test \
        --pred_dir my_outputs \
        --gt_dir ground_truth_test

Filename convention:
    ground_truth_test/001.png          <- ground truth
    noise_test/001_noise.png            <- noisy input (given to you)
    my_outputs/001.png                   <- YOUR reconstruction (same id, NO suffix)

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def ssim_value(a: np.ndarray, b: np.ndarray) -> float:
    return float(sk_ssim(
        a, b,
        channel_axis=-1,
        data_range=1.0,
        win_size=7,
        gaussian_weights=False,
        use_sample_covariance=True,
        K1=0.01,
        K2=0.03,
    ))


def normalize_delta_psnr(delta_psnr: float) -> float:
    return float(np.clip(delta_psnr / 15.0, 0.0, 1.0))


def official_composite(delta_psnr: float, delta_ssim: float) -> float:
    normalized_delta_psnr = normalize_delta_psnr(delta_psnr)
    positive_delta_ssim = max(delta_ssim, 0.0)
    return float(0.6 * normalized_delta_psnr + 0.4 * positive_delta_ssim)


def load_rgb_float(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise SystemExit(f"ERROR: cannot read image {path}: {exc}")
    return arr


def find_image_ids(gt_dir: Path) -> list[str]:
    ids = sorted(p.stem for p in gt_dir.glob("*.png"))
    if not ids:
        raise SystemExit(f"ERROR: no .png files found in ground-truth dir {gt_dir}")
    return ids


def evaluate_image(image_id: str, pred_path: Path, gt_path: Path, noisy_path: Path) -> dict:
    pred = load_rgb_float(pred_path)
    gt = load_rgb_float(gt_path)
    noisy = load_rgb_float(noisy_path)

    if pred.shape != gt.shape:
        raise SystemExit(f"ERROR: {image_id}: your prediction shape {pred.shape} "
                          f"doesn't match ground truth shape {gt.shape}")
    if noisy.shape != gt.shape:
        raise SystemExit(f"ERROR: {image_id}: noisy input shape {noisy.shape} "
                          f"doesn't match ground truth shape {gt.shape}")

    psnr = float(sk_psnr(gt, pred, data_range=1.0))
    ssim = ssim_value(gt, pred)
    noisy_psnr = float(sk_psnr(gt, noisy, data_range=1.0))
    noisy_ssim = ssim_value(gt, noisy)
    delta_psnr = psnr - noisy_psnr
    delta_ssim = ssim - noisy_ssim
    composite = official_composite(delta_psnr, delta_ssim)

    return {
        "image_id": image_id,
        "psnr": psnr,
        "ssim": ssim,
        "noisy_psnr": noisy_psnr,
        "noisy_ssim": noisy_ssim,
        "delta_psnr": delta_psnr,
        "delta_ssim": delta_ssim,
        "composite_score": composite,
    }


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--noisy_dir", required=True, type=Path,
                     help="Folder of noisy input images given to you")
    ap.add_argument("--pred_dir", required=True, type=Path,
                     help="Folder of YOUR reconstructed images")
    ap.add_argument("--gt_dir", required=True, type=Path,
                     help="Folder of ground-truth images you have access to "
                          "(e.g. your own held-out validation split)")
    args = ap.parse_args()

    for label, d in [("--noisy_dir", args.noisy_dir), ("--pred_dir", args.pred_dir),
                      ("--gt_dir", args.gt_dir)]:
        if not d.exists() or not d.is_dir():
            raise SystemExit(f"ERROR: {label} does not exist or is not a folder: {d}")

    image_ids = find_image_ids(args.gt_dir)

    rows, missing = [], []
    for image_id in image_ids:
        gt_path = args.gt_dir / f"{image_id}.png"
        noisy_path = args.noisy_dir / f"{image_id}_noise.png"
        pred_path = args.pred_dir / f"{image_id}.png"

        if not noisy_path.exists():
            missing.append(str(noisy_path))
            continue
        if not pred_path.exists():
            missing.append(str(pred_path))
            continue

        rows.append(evaluate_image(image_id, pred_path, gt_path, noisy_path))

    if missing:
        print("The following expected files were not found:")
        for m in missing:
            print(f"  - {m}")
        print(f"\nScored {len(rows)} / {len(image_ids)} images.\n")

    if not rows:
        raise SystemExit("ERROR: no images could be scored.")

    mean_psnr = float(np.mean([r["psnr"] for r in rows]))
    mean_ssim = float(np.mean([r["ssim"] for r in rows]))
    mean_delta_psnr = float(np.mean([r["delta_psnr"] for r in rows]))
    mean_delta_ssim = float(np.mean([r["delta_ssim"] for r in rows]))
    mean_composite = float(np.mean([r["composite_score"] for r in rows]))

    print("=" * 46)
    print("EVALUATION COMPLETE")
    print("=" * 46)
    print(f"Images scored:      {len(rows)}")
    print(f"Mean PSNR:           {mean_psnr:.4f} dB")
    print(f"Mean SSIM:           {mean_ssim:.6f}")
    print(f"Mean Delta PSNR:     {mean_delta_psnr:+.4f} dB")
    print(f"Mean Delta SSIM:     {mean_delta_ssim:+.6f}")
    print()
    print(f"Composite Score:     {mean_composite:.8f}")
    print("=" * 46)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

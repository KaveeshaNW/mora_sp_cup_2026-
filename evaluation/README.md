# Evaluation Scripts

This folder contains the participant-side evaluation script for the competition.

## Files

- `evaluate.py` — Computes the denoising score (PSNR, SSIM, and composite score) for your images.

## Install

Install the dependencies required by `evaluate.py` from the repository root:

```bash
pip install -r evaluation/requirements.txt
```

The required packages are `numpy`, `Pillow`, and `scikit-image`.

## Evaluate

`evaluate.py` computes your denoising score on image pairs for which ground truth is available.

Run it with three folders: the noisy input, your denoised output, and the corresponding ground truth:

```bash
python evaluation/evaluate.py --noisy_dir competition_data/public/noisy --pred_dir competition_data/public/denoised --gt_dir competition_data/public/ground_truth
```

- `--noisy_dir`: folder containing the noisy input images.
- `--pred_dir`: folder containing your denoised (`<id>.png`) images.
- `--gt_dir`: folder containing the corresponding ground-truth images.

## Filename Convention

Your denoised output must follow the required convention: use the same image ID as the ground truth without the `_noise` suffix.

```text
ground_truth/001.png          <- ground truth
noisy/001_noise.png           <- noisy input
denoised/001.png              <- your denoised output
```

## Output

Running the script prints:

- Mean PSNR
- Mean SSIM
- Mean Delta PSNR
- Mean Delta SSIM
- Composite Score

> **Note:** This script is provided for participant-side self-evaluation.
> It implements the same PSNR, SSIM, and composite-score calculation used
> for official scoring. The organizer-side evaluator additionally applies
> strict submission validation, including file count, filenames, PNG/RGB
> format, image dimensions, and other competition requirements.

For the complete evaluation and submission rules, refer to the official competition document.

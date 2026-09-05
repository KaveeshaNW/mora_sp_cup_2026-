# Classical Denoising Baseline

## Installation

Install the required dependencies from the repository root:

```bash
pip install -r baseline/requirements.txt
```

## How to denoise

Run the classical pipeline (defect-pixel correction + Non-Local Means) with:

```bash
python baseline/denoise.py --input_dir competition_data/public/noisy --output_dir competition_data/public/denoised
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--denoise_method` | `nlm` | Denoising algorithm: `nlm` (non-local means), `bilateral`, or `none`. |
| `--nlm_h` | `10.0` | NLM denoising strength. Higher values remove more noise but can soften detail. |
| `--skip_defect_correction` | off | Skip the defect-pixel (outlier) correction step. |

Example with non-local means strength set to `15`:

```bash
python baseline/denoise.py --input_dir competition_data/public/noisy --output_dir competition_data/public/denoised --nlm_h 15
```

## How to evaluate

Score your denoised output against ground truth using the contestant self-check evaluator:

```bash
python evaluation/evaluate.py --noisy_dir competition_data/public/noisy --pred_dir competition_data/public/denoised --gt_dir competition_data/public/ground_truth
```

## Baseline results (public 460-image set)

```text
==============================================
EVALUATION COMPLETE
==============================================
Images scored:      460
Mean PSNR:           24.3360 dB
Mean SSIM:           0.671659
Mean Delta PSNR:     +4.5103 dB
Mean Delta SSIM:     +0.207275

Composite Score:     0.26329211
==============================================
```

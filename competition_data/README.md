# competition_data

This folder holds the competition datasets for the low-light image denoising challenge.

## Directory Layout

```text
competition_data/
├── public/
│   ├── ground_truth/
│   │   ├── 001.png
│   │   ├── ...
│   │   └── 460.png
│   ├── noisy/
│   │   ├── 001_noise.png
│   │   ├── ...
│   │   └── 460_noise.png
│   └── denoised/
│
└── submissions/
    ├── noisy/
    │   ├── 461_noise.png
    │   ├── ...
    │   └── 480_noise.png
    └── denoised/
        ├── 461.png
        ├── ...
        └── 480.png
```

## How to Run the Baseline

Place the provided dataset in the folders shown above.

### 1. Denoise the Public Images

From the repository root, run:

```bash
python baseline/denoise.py --input_dir competition_data/public/noisy --output_dir competition_data/public/denoised
```

### 2. Evaluate the Denoised Outputs

Evaluate the generated outputs against the available public ground truth:

```bash
python evaluation/evaluate.py --noisy_dir competition_data/public/noisy --pred_dir competition_data/public/denoised --gt_dir competition_data/public/ground_truth
```

The evaluation script is provided for participant-side self-evaluation on images for which ground truth is available.

## Preliminary-Round Submission Convention

For the hidden preliminary-round images:

```text
461_noise.png
462_noise.png
...
480_noise.png
```

the corresponding denoised outputs must be saved as:

```text
461.png
462.png
...
480.png
```

inside:

```text
competition_data/submissions/denoised/
```

For the official preliminary-round image submission, package exactly these 20 denoised PNG files into:

```text
TeamName.zip
```

The report must be submitted separately as:

```text
TeamName_Report.pdf
```

The team's complete solution code must be maintained in the team's private GitHub repository according to the official competition submission rules.

For complete submission requirements, repository requirements, and final-round rules, refer to the official competition document.

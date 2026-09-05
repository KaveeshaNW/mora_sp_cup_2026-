# Welcome to the Low-Light Image Denoising & Enhancement Challenge

**Mora SP Cup 2026**

## About the Challenge

Low-light photography is challenging because camera sensors become more
susceptible to noise and other artifacts under limited illumination.

In this competition, participants will work with a custom dataset prepared
from clean mobile-phone photographs and corrupted using a synthetic low-light
noise model that reproduces several common sensor-related artifacts.

Your task is to design an effective denoising solution that removes these
artifacts while preserving image details, structure, and visual quality.

Whether you are new to image processing or testing novel approaches, this is
a hands-on opportunity to solve a practical digital imaging problem.

## The Task

- Denoise a collection of noisy images.
- The dataset contains different severity levels of noise.
- All images have a resolution of **992 × 992 pixels**.
- The complete dataset consists of 500 noisy images and 500 corresponding
  ground-truth images, divided as follows:

  - **Public set:** 460 noisy images
    (`001_noise.png`–`460_noise.png`) together with their ground-truth
    images (`001.png`–`460.png`). Use these images to develop, train,
    and validate your approach.

  - **Preliminary submission set:** 20 noisy images
    (`461_noise.png`–`480_noise.png`) are provided without their
    corresponding ground-truth images. Teams must denoise these images
    and submit the outputs as `461.png`–`480.png`.

  - **Final-round set:** A further 20 image pairs (`481`–`500`) are kept
    fully hidden by the organizers. Finalists will receive only the
    corresponding noisy images during the physical final round.

- The provided noisy images must **not** be renamed.

## Project Layout

Place the images in the correct directories before running the baseline:

```text
competition_data/
|-- public/
|   |-- ground_truth/
|   |   |-- 001.png
|   |   |-- ...
|   |   `-- 460.png
|   |
|   |-- noisy/
|   |   |-- 001_noise.png
|   |   |-- ...
|   |   `-- 460_noise.png
|   |
|   `-- denoised/
|
`-- submissions/
    |-- noisy/
    |   |-- 461_noise.png
    |   |-- ...
    |   `-- 480_noise.png
    |
    `-- denoised/

import numpy as np
import bm3d
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import estimate_sigma
import os

# 1. Load your noisy image
# Replace this with the actual path to your file (e.g., "C:/images/dark_room.jpg")
file_path = r"C:\Users\Kaveesha\Desktop\003_noise.png"

if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}")
    exit()

# Convert to float (values between 0.0 and 1.0) which BM3D requires
image = img_as_float(io.imread(file_path))

# 2. Estimate the unknown noise level
# channel_axis=-1 tells the estimator it is a color image (RGB)
# average_sigmas=True combines the R, G, and B noise estimates into one number for BM3D
sigma_est = estimate_sigma(image, channel_axis=-1, average_sigmas=True)
print(f"Estimated Noise Standard Deviation: {sigma_est:.4f}")

# 3. Apply the BM3D algorithm
print("Running BM3D... (This may take a while for large images)")
# Change .bm3d to .bm3d_rgb
denoised_image = bm3d.bm3d_rgb(image, sigma_psd=sigma_est)

# Ensure pixel values don't accidentally exceed standard bounds during math operations
denoised_image = np.clip(denoised_image, 0, 1)

# 4. Save the cleaned image back to your computer
output_path = r"C:\Users\Kaveesha\Desktop\denoised_result3.jpg"
# Convert back to standard 8-bit image format before saving
io.imsave(output_path, img_as_ubyte(denoised_image))
print(f"Success! Saved clean image to {output_path}")
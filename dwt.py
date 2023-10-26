import pywt
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load an image from 'D:\Downloads' directory
image_path = r'D:\Downloads\zoro-pfp-0v1nn9tjqh71x5fp.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Choose a wavelet and its decomposition level
wavelet = 'haar'
level = 2

# Perform DWT on the image
coeffs = pywt.wavedec2(image, wavelet, level=level)

# Quantization (for simplicity, we'll use uniform quantization)
quantization_step = 10

# Quantize each array within the tuple
quantized_coeffs = [np.round(c / quantization_step) if not isinstance(c, tuple) else tuple(np.round(sub_c / quantization_step) for sub_c in c) for c in coeffs]

# Reconstruct the coefficients into the correct format
reconstructed_coeffs = [quantized_coeffs[0]] + quantized_coeffs[1:]

# Inverse DWT to reconstruct the compressed image
reconstructed_image = pywt.waverec2(reconstructed_coeffs, wavelet)

# Resize the reconstructed image to match the size of the original image
reconstructed_image = cv2.resize(reconstructed_image, (image.shape[1], image.shape[0]))

# Calculate the absolute difference between the original and reconstructed images
diff_image = cv2.absdiff(image, reconstructed_image.astype(np.uint8))

# Display the original and reconstructed images side by side
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_image.astype(np.uint8), cmap='gray')
plt.title('Reconstructed Image (Compressed)')
plt.axis('off')

# Display the difference image
plt.subplot(1, 3, 3)
plt.imshow(diff_image, cmap='gray')
plt.title('Absolute Difference Image')
plt.axis('off')

plt.show()

# Save the reconstructed image
reconstructed_image_path = r'D:\Downloads\reconstructed_image1.jpg'
cv2.imwrite(reconstructed_image_path, reconstructed_image.astype(np.uint8))

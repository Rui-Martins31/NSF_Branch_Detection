# More Information:
# https://github.com/facebookresearch/sam2?tab=readme-ov-file

# Imports
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#
from utils import show_points, show_masks, distance

np.random.seed(3)

# Device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Image
image = Image.open("./images/tree_03.jpg")
image = np.array(image.convert("RGB"))

# Compute image center
image_width  = len(image)
image_length = len(image[0])
print(f"Image Size (px): {image_length}x{image_width}")
image_center = np.array([round(image_length/2), round(image_width/2)])
print(f"Image Center (px): ({image_center[0]}, {image_center[1]})")

input_point = np.array([[800, 500]])
input_label = np.array([1])
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.scatter(image_center[0], image_center[1])
plt.axis('on')
plt.show()  

# Predict mask
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg  = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor  = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float16
with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

# Get centroid
for mask in masks:
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # Try to smooth contours
    contours    = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]

    # Moments
    M = cv2.moments(contours[0])

    # Centroid (cx, cy)
    if M['m00'] != 0: # Avoid division by zero
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        print(f"Centroid (cX, cY): ({cX}, {cY})")

# Get distance
dist_ = distance(image_center, np.asarray([cX, cY]))
print(f"Distance between the mask centroid and the image center (px): {dist_}")

# DEBUG - Plot
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.scatter(image_center[0], image_center[1])
plt.scatter(cX, cY)
plt.axis('on')
plt.show()  
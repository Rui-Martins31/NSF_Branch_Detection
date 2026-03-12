import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import show_points, show_masks, show_box, distance
from sam import predict_mask, compute_mask_center
from grounding_dino import grounding_dino_detect

# Image
IMAGE_PATH   = "./images/tree_branch_03"
IMAGE_RESIZE = (400, 300)
image_pil    = Image.open(f"{IMAGE_PATH}.png").resize(IMAGE_RESIZE)
image        = np.array(image_pil.convert("RGB"))

# Image center
image_width  = len(image)
image_length = len(image[0])
image_center = np.array([round(image_length / 2), round(image_width / 2)])
print(f"Image Size (px): {image_length}x{image_width}")
print(f"Image Center (px): ({image_center[0]}, {image_center[1]})")

# Grounding DINO detection
dino_box, dino_label, dino_score = grounding_dino_detect(image_pil = image_pil)

# Bounding box center as SAM point
cx = (dino_box[0] + dino_box[2]) / 2
cy = (dino_box[1] + dino_box[3]) / 2
input_point = np.array([[cx, cy]])
input_label = np.array([1])

# DEBUG
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image_pil)
show_box(dino_box, ax)
show_points(input_point, input_label, ax)
ax.scatter(image_center[0], image_center[1], c="yellow", s=200, zorder=5)
ax.axis("on")
plt.title(f"{dino_label} ({dino_score:.2f})")
plt.show()

# SAM segmentation
masks, scores, logits = predict_mask(image, input_point, input_label)
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
cX, cY = compute_mask_center(masks=masks)

# Distance
dist_ = distance(image_center, np.array([cX, cY]))
print(f"Distance between mask centroid and image center (px): {dist_}")

# DEBUG - Final plot
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.scatter(image_center[0], image_center[1], c="yellow", s=200, zorder=5, label="Image center")
plt.scatter(cX, cY, c="red", s=200, zorder=5, label="Mask centroid")
plt.legend()
plt.axis("on")
plt.show()

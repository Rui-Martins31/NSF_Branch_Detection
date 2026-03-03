import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator

from utils import show_points, show_masks, show_box, distance
from sam import predict_mask

np.random.seed(3)

# Device
device = Accelerator().device

# Image
image_path = "./images/tree_03"
image_pil  = Image.open(f"{image_path}.jpg")
image      = np.array(image_pil.convert("RGB"))

# Image center
image_width  = len(image)
image_length = len(image[0])
print(f"Image Size (px): {image_length}x{image_width}")
image_center = np.array([round(image_length / 2), round(image_width / 2)])
print(f"Image Center (px): ({image_center[0]}, {image_center[1]})")

# Grounding DINO detection
model_id    = "IDEA-Research/grounding-dino-tiny"
text_labels = [["tree branch"]]

processor  = AutoProcessor.from_pretrained(model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

inputs = processor(images=image_pil, text=text_labels, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = dino_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image_pil.size[::-1]]
)

result = results[0]
if len(result["boxes"]) == 0:
    raise RuntimeError("No objects detected. Try lowering the detection threshold.")

# Pick highest-confidence detection
best_idx = result["scores"].argmax().item()
box      = result["boxes"][best_idx].tolist()
label    = result["labels"][best_idx]
score    = result["scores"][best_idx]
print(f"\nDetected '{label}' with confidence {score:.3f} at {[round(x, 2) for x in box]}")

# Bounding box center as SAM point prompt
cx = (box[0] + box[2]) / 2
cy = (box[1] + box[3]) / 2
input_point = np.array([[cx, cy]])
input_label = np.array([1])

# DEBUG - Detection + prompt point
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image_pil)
show_box(box, ax)
show_points(input_point, input_label, ax)
ax.scatter(image_center[0], image_center[1], c="yellow", s=200, zorder=5)
ax.axis("on")
plt.title(f"{label} ({score:.2f})")
plt.show()

# SAM segmentation
masks, scores, logits = predict_mask(image, input_point, input_label, device=str(device))
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

# Mask centroid
for mask in masks:
    mask      = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours  = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]

    M = cv2.moments(contours[0])
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        ys, xs = np.where(mask)
        cX, cY = int(xs.mean()), int(ys.mean())
    print(f"Centroid (cX, cY): ({cX}, {cY})")

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

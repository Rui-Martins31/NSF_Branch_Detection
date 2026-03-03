import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import cv2
import numpy as np

np.random.seed(3)

def predict_mask(image, input_point, input_label,
                 checkpoint="./checkpoints/sam2.1_hiera_small.pt",
                 model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
                 device=None):
    # Device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Predict
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float16

    with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

    return masks, scores, logits

def compute_mask_center(
    masks
):
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

    return cX, cY

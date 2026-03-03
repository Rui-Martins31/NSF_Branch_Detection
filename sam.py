import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def predict_mask(image, input_point, input_label,
                 checkpoint="./checkpoints/sam2.1_hiera_small.pt",
                 model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
                 device=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

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

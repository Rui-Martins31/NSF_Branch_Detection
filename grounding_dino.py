import torch
from PIL import Image
from torchvision.ops import box_convert

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

MODEL_CONFIG  = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
TEXT_PROMPT   = "tree branch"
BOX_TRESHOLD  = 0.35
TEXT_TRESHOLD = 0.25

model = load_model(MODEL_CONFIG, MODEL_WEIGHTS)

def grounding_dino_detect(
    image_pil: Image.Image,
    text_prompt: str        = TEXT_PROMPT,
    box_threshold: float    = BOX_TRESHOLD,
    text_threshold: float   = TEXT_TRESHOLD,
    device: str             = None
):
    # Device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Preprocess PIL image
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    if len(boxes) == 0:
        raise RuntimeError("No objects detected. Try lowering the detection threshold.")

    # Convert normalized cxcywh -> xyxy pixel coords
    w, h = image_pil.size
    boxes_pixel = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy")

    # Pick highest-confidence detection
    best_idx = logits.argmax().item()
    box      = xyxy[best_idx].tolist()
    label    = phrases[best_idx]
    score    = logits[best_idx].item()
    print(f"\nDetected '{label}' with confidence {score:.3f} at {[round(x, 2) for x in box]}")

    return box, label, score
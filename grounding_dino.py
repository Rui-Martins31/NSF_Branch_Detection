from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator

# Constants
MODEL_ID: str                = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT: list[list[str]] = [["tree branch"]]

def grounding_dino_detect(
    image_pil: Image,
    model_id: str                = MODEL_ID,
    text_labels: list[list[str]] = TEXT_PROMPT
):
    # Device
    device = Accelerator().device    

    # Processor and model
    processor  = AutoProcessor.from_pretrained(model_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Inputs
    inputs = processor(images=image_pil, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    # Results
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

    return box, label, score
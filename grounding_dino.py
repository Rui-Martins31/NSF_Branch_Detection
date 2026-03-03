import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator
from utils import show_box

model_id = "IDEA-Research/grounding-dino-tiny"
device = Accelerator().device

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_path  = "./images/tree_02"
image       = Image.open(f"{image_path}.jpg")
text_labels = [["tree branch"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

# Plot result
result = results[0]

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)

for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
    box = box.tolist()
    show_box(box, ax)
    x0, y0 = box[0], box[1]
    ax.text(x0, y0 - 5, f"{label} {score:.2f}", color="white", fontsize=10,
            bbox=dict(facecolor="green", alpha=0.6, pad=2))
    print(f"\nDetected {label} with confidence {round(score.item(), 3)} at location {[round(x, 2) for x in box]}")

plt.axis("off")
plt.savefig(f"{image_path}_detections.jpg", bbox_inches="tight", dpi=150)
plt.show()
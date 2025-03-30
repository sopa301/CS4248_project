from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
image_path = "/home/andrew/CS4248_project/dataset/noto-emoji/png/512/emoji_u1f923.png"

image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs)

print("Keys in output:", output.keys())  # Check available attributes
print("Logits:", output.logits.shape)
model.head = torch.nn.Identity()

with torch.no_grad():
    output = model(**inputs)

print("Output after removing head:", output)
print("Logits:", output.logits.shape)

features = model.forward_features(inputs['pixel_values'])  # [1, seq_len, hidden_dim]
pooled = features.mean(dim=1)                              # [1, hidden_dim]

print("Pooled", pooled.shape)
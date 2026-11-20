
import torch
from precompute_pooled_features import load_uni2

print("CUDA available:", torch.cuda.is_available(), flush=True)

model = load_uni2(pretrained=True)
model = model.eval().cuda()

print("Pretrained UNI2-h loaded successfully!", flush=True)
print("Model device:", next(model.parameters()).device, flush=True)

with torch.inference_mode():
    x = torch.randn(1, 3, 224, 224, device="cuda")
    y = model(x)

print("Output shape:", tuple(y.shape), flush=True)

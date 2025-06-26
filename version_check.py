import torch, onnxruntime as ort, platform

print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("ORT device:", ort.get_device())
print(
    "GPU available:",
    torch.cuda.is_available(),
    "| 이름:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
)

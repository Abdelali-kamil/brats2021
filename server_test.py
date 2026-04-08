import torch
import sys

print("-" * 30)
print("SERVER DIAGNOSTICS")
print("-" * 30)

if torch.cuda.is_available():
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    print("❌ NO GPU FOUND")

print("-" * 30)
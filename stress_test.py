import torch
import time

def run_stress_test():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    device = torch.device('cuda:0')
    print(f"🔥 Starting MAX Stress Test on: {torch.cuda.get_device_name(device)}")
    
    # Allocate massive tensors to fill up VRAM
    # A 25,000 x 25,000 matrix takes several gigabytes
    matrix_size = 25000 
    
    print("Allocating memory...")
    a = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
    b = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)

    print("Warming up GPU...")
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    print("🚀 Pushing to Absolute Maximum! (Press Ctrl+C to stop)")
    
    # Infinite loop of heavy compute to spike power draw and utilization to 100%
    try:
        while True:
            c = torch.matmul(a, b)
            # Delete the result immediately so memory doesn't overflow, just burns compute
            del c 
    except KeyboardInterrupt:
        print("\nStress test stopped. Cooling down...")

if __name__ == "__main__":
    run_stress_test()
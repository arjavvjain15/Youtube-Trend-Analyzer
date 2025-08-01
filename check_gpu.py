# check_gpu.py
import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"✅ Success! Found {gpu_count} GPU(s) available.")
    
    # Get and print the name of the current GPU
    current_gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu_index)
    print(f"Using GPU #{current_gpu_index}: {gpu_name}")
else:
    print("❌ Failure. PyTorch cannot find a CUDA-enabled GPU.")
    print("The training will run on the CPU.")
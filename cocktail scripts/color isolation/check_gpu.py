import torch

if torch.cuda.is_available():
    print("YOLO is running on GPU")
else:
    print("YOLO is running on CPU")

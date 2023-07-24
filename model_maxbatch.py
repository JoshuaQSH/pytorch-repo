import torchvision.models as models
import torch
import numpy as np

# optimal_batch_size
model = models.resnet18()
device = torch.device("cuda:0")
model.to(device)
optimal_batch_size = 1

dummy_input = torch.randn(optimal_batch_size, 3, 224, 224, dtype=torch.float).to(device)
_ = model(dummy_input)
print("Done Infer")
while(optimal_batch_size):
    dummy_input = torch.randn(optimal_batch_size, 3, 224, 224, dtype=torch.float).to(device)
    try:
        _ = model(dummy_input)
        optimal_batch_size += 1
    except Exception as err:
        print("Optimal batch size is: ", optimal_batch_size)
        raise err

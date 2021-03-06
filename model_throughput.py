import torchvision.models as models
import torch
import numpy as np
from timm.models import create_model

# Throughput counting below
model_name = ['squeezenet1_0', 'densenet_161', 'resnet18',
                   'vgg11', 'shufflenet_v2_x1_0',
                   'mobilenet_v3_small', 'mnasnet1_0',
                   'inception_v3', 'googlenet', 'alexnet']

# torchvisions model
model = models.resnet18()

# timm models ConvNeXt for examble
model_name_conv = ['convnext_small']
model = create_model(
	'convnext_small',
	pretrained=False,
	num_classes=1000,
	drop_path_rate=0,
	head_init_scale=1.0)

device = torch.device("cuda:1")
model.to(device)
dummy_input = torch.randn(306, 3, 224, 224, dtype=torch.float).to(device)
repetitions = 100
total_time = 0
with torch.no_grad():
    for rep in range(repetitions):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)/1000
        total_time += curr_time
# optimal_batch_size here, should be tested by the max batch-size
optimal_batch_size = 64
throughput = (repetitions*optimal_batch_size) / total_time
print("Model: ", model_name_conv[0])
print("Final Throughput: ", throughput)
print(model)

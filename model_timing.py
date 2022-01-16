import torchvision.models as models
import torch
import numpy as np
import time


model_name = ['squeezenet1_0', 'densenet_161', 'resnet18',
                   'vgg11', 'shufflenet_v2_x1_0',
                   'mobilenet_v3_small', 'mnasnet1_0',
                   'inception_v3', 'googlenet', 'alexnet']
models_res = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

device = torch.device("cuda:0")
experts_res = []
for _mn in models_res:
    experts_res.append(getattr(models, _mn)().to(device))
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True),
repetitions = 300
timings = np.zeros((repetitions,1))


print("GPU Warm-up ...", device)
# GPU warm up
for _ in range(10):
    _ = experts_res[0](dummy_input)


print("Begin ...(GPU)")
# measurement
with torch.no_grad():
    for index, model in enumerate(experts_res):
        for rep in range(repetitions):
            starter.record()
            with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False) as prof:
                _ = model(dummy_input)
            ender.record()

            # wait for sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        
        # Output the Json file for further discoving
        # open in chrome://tracing
        prof.export_chrome_trace('./profile/{}_profile.json'.format(models_res[index]))

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print("Time (GPU): {:.4f}ms, Model: {}".format(mean_syn, models_res[index]))


print("Begin ...(CPU)")
experts_res_cpu = []
# CPU version
for _mn in models_res:
    experts_res_cpu.append(getattr(models, _mn)().cpu())
dummy_input = dummy_input.cpu()

timing_cpu = np.zeros((repetitions, 1))
with torch.no_grad():
    for index, model in enumerate(experts_res_cpu):
        for rep in range(repetitions):
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            timing_cpu[rep] = end - start
        mean_syn = np.sum(timing_cpu) / repetitions
        std_syn = np.std(timing_cpu)
        print("Time (CPU): {:.4f}ms, Model: {}".format(mean_syn*1000, models_res[index]))


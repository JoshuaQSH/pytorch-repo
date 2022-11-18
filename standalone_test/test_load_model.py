import torch
import torchvision.models as models
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = models.resnet18()
load_model = torch.load("../checkpoint/vgg16_ckpt.pth")
load_model = load_model.module
# model.load_state_dict(m_stat_dict)
# print(load_model.cpu())

loaded_dict = torch.load("../checkpoint/vgg16_stat.pth")
loaded_model_stat = torch.nn.DataParallel(model).cuda()
loaded_model_stat.state_dict = loaded_dict
loaded_model_stat = loaded_model_stat.cpu()
#  torch.save(loaded_model_stat.state_dict(), "./checkpoint/resnet18_cpu_stat_2.pt")

load_model = load_model.cpu()
torch.save(load_model, "../checkpoint/vgg16_cpu_whole.pt")
torch.save(load_model.state_dict(), "../checkpoint/vgg16_cpu_stat.pt")

print("Done")

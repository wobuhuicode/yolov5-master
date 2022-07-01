from thop import profile
import torch

from models.experimental import attempt_load

# 计算模型的 flops

weights_path = "weights/ghost_relu3.pt"

model = attempt_load(weights_path, device='cpu')
input = torch.randn(1, 3, 640, 640)
flops, params = profile(model, inputs=(input, ))
print('flops:', flops)
print('params:', params)
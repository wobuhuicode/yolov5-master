from copy import deepcopy
import time
import numpy as np
import torch
import thop
from tqdm import tqdm
from utils.torch_utils import print_model_details
import prettytable as pt

# 按 yaml 初始化模型
from models.yolo import Detect, Model
cfg = '/mnt21t/home/wyh/zkl_project/yolov5-master/models/nas_1.yaml'
model = Model(cfg, ch=3, nc=1)
model.fuse()
model.eval()
model.info()
dummy_input = torch.randn(1, 3, 640, 640)
model(dummy_input)
flops_, params_ = thop.profile(model, inputs=(dummy_input,), verbose=True)
print(f"FLOPs: {flops_ / 1E9 * 2} G, Params: {params_ / 1E6} M")

for k, m in model.named_modules():
    if isinstance(m, Detect):
        print(f"{k}: {m}")
        m.inplace = True
        m.onnx_dynamic = False
        m.export = True

f = 'nas_1.onnx'

torch.onnx.export(
    model,
    dummy_input,
    f,
    verbose=False,
    opset_version=12,
    input_names=['images'],
    output_names=['output'],
    )

import onnx
import onnxsim

# Checks
model_onnx = onnx.load(f)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model
model_onnx, check = onnxsim.simplify(model_onnx,
                                        dynamic_input_shape=False,
                                        input_shapes=None)
assert check, 'assert check failed'
onnx.save(model_onnx, f)

input('Plase press enter to continue...')

## 按行添加数据
tb = pt.PrettyTable()
tb.field_names = ["Model", "FLOPs(G)", "Mean Inference Time(ms)", "CUDA Memory Cache(MB)"]

weights = ['weights/fbnet.pt', 'weights/md.pt', 'weights/ofa-bdd.pt', 'weights/ghost_relu3.pt']
model_names = ['fbnet', 'mobiledets', 'ofa', 'ours']
flops = []
params = []
times = []
mems = []

device='cuda:0'

dummy_input = torch.randn(1, 3, 640, 640).cuda()
repetitions = 3000


for i in range(len(model_names)):
    print("===========================================================")
    print("Profiling model: ", model_names[i])
    if model_names[i] != 'ours':
        print_model_details(model_names[i])
    from models.experimental import attempt_load
    model = attempt_load(weights[i], device=device, inplace=True, fuse=True)
    model.eval()
    model.info()

    # warm-up
    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...')
    with torch.no_grad():
        for _ in range(200):
            _ = model(dummy_input)

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    torch.cuda.synchronize()

    print('testing ...')
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time
    
    # 计算平均时间
    mean_time = np.mean(timings)
    times.append(mean_time)
    print('mean time: ', mean_time, 'ms')

    # time_start = time.time()
    # for i in range(10000):
    #     predict = model(input)
    # torch.cuda.synchronize()
    # time_end = time.time()
    # time_sum = time_end - time_start
    # times.append(time_sum / 10)

    flops_, params_ = thop.profile(deepcopy(model), inputs=(dummy_input,), verbose=False)
    flops.append(flops_ * 2 / 1E9)
    params.append(params_ / 1E6)
    mems.append(torch.cuda.memory_reserved(device=device) / 1E6)
    torch.cuda.empty_cache()


for i in range(len(model_names)): 
    tb.add_row([model_names[i], '%.2g' % flops[i], '%.3g' % times[i], '%.3g' % mems[i]])


print(tb)

# with profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CUDA,
#     ],
#     with_stack=False, with_flops=True, profile_memory=True) as p:
#     for i in range(1000):
#         out = model(input)

# print(p.key_averages().table(
#     sort_by="self_cuda_time_total", row_limit=-1))

# CUDA_VISIBLE_DEVICES=1 python bdd_profile.py
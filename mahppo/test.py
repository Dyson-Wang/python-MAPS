import torch

# 检查GPU是否可用
is_cuda_available = torch.cuda.is_available()
print(f"CUDA 可用: {is_cuda_available}")

if is_cuda_available:
    # 获取可用的GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU 数量: {gpu_count}")

    # 获取每个GPU的名称
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i} 名称: {gpu_name}")

    # 获取当前使用的GPU索引
    current_device = torch.cuda.current_device()
    print(f"当前使用的GPU索引: {current_device}")

    # 获取当前GPU的名称
    current_gpu_name = torch.cuda.get_device_name(current_device)
    print(f"当前GPU名称: {current_gpu_name}")

    # 创建一个张量并移动到GPU上
    x = torch.randn(5, 5).to(current_device)
    print(f"张量x所在的设备: {x.device}")
else:
    print("没有可用的GPU。")
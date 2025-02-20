import inspect
import torch
import torch.nn as nn
from . import config, MainWindow
from .export import TorchModel

# 读取模型函数
def load_model(path, print_model = False):
    # 添加加载模型需要的层
    nn_core_classes = [cls for _, cls in inspect.getmembers(nn.modules, inspect.isclass)]   # 获取PyTorch的核心层
    custom_classes = [getattr(config, name) for name in dir(config) if isinstance(getattr(config, name), type)]  # 获取config的自定义类
    torch.serialization.add_safe_globals(nn_core_classes)   # PyTorch核心层
    torch.serialization.add_safe_globals(custom_classes)    # 自定义类
    torch.serialization.add_safe_globals([TorchModel, set]) # 其他层

    # 加载模型
    model = torch.load(path, weights_only=True)
    if print_model:
        print(model)
    return model

def start():
    MainWindow.start()


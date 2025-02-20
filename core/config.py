import sys
import numpy as np
import json
import networkx as nx
import inspect
import torch
import torch.nn as nn
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSplitter, QListWidget, QListWidgetItem
from PyQt5.QtWidgets import QMenuBar, QMenu, QAction, QFileDialog, QMessageBox, QComboBox, QStyledItemDelegate, QScrollBar
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsTextItem, QGraphicsDropShadowEffect, QGraphicsPathItem, QGraphicsLineItem
from PyQt5.QtWidgets import QDialog, QLabel, QFormLayout, QLineEdit, QPushButton, QTextEdit
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPen, QBrush, QColor, QFont, QLinearGradient, QPainterPath, QPolygonF, QPainterPathStroker, QPainter,  QWheelEvent, QMouseEvent, QTransform
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer


#region UI参数
canvas_size = 5000  # 画布大小
grid_size = 17  # 网格大小
zoom_factor = 1.2  # 缩放因子
min_zoom = 0.35  # 最小缩放比例
max_zoom = 2  # 最大缩放比例
module_width = grid_size * 10  # 模块矩形网格宽
module_height = grid_size * 6   # 模块矩形网格高
red_dot_edge = grid_size * 0.7     # 红点边长
anchor_label_size = 6     # "×" 大小
arrow_size = 20     # 箭头大小

coord_label_width = 140  # 坐标标签宽度
coord_label_height = 20   # 坐标标签高度
corner_radius = grid_size / 3 # 圆角半径
dragging_time = 100   # 长按检测时间
#endregion


#region 张量操作相关类定义

# 逐元素相加
class Add(nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)   

# 逐元素相乘
class Mul(nn.Module):
    def forward(self, x, y):
        return torch.mul(x, y)  

# 逐元素求最大值
class Maximum(nn.Module):
    def forward(self, x, y):
        return torch.maximum(x, y)   
    
# 逐元素求最小值
class Minimum(nn.Module):
    def forward(self, x, y):
        return torch.minimum(x, y)  

# 指定维度求均值
class Mean(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = None
        self.keepdim = False
    
    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)  

# 指定维度求和
class Sum(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = None
        self.keepdim = False

    def forward(self, x):
        return torch.sum(x, dim=0, keepdim=True)  
    
# 张量拼接
class Cat(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 0
    
    def forward(self, *tensors):
        return torch.cat(tensors, dim=self.dim)

# 张量堆叠
class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 0
    
    def forward(self, *tensors):
        return torch.stack(tensors, dim=self.dim)

# 张量拆分
class Split(nn.Module):
    def __init__(self):
        super().__init__()
        self.split_size = 2
        self.dim = 0
    
    def forward(self, x):
        return torch.split(x, self.split_size, dim=self.dim)

# 张量重复
class Repeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.repeats = (1,1)
    
    def forward(self, x):
        return x.repeat(*self.repeats)

# 视图变换
class View(nn.Module):
    def __init__(self):  
        super().__init__()
        self.shape = (-1,)

    def forward(self, x):
        return x.contiguous().view(self.shape)

# 多个维度重排
class Permute(nn.Module):
    def __init__(self):  
        super().__init__()
        self.dims = (0, 1, 2)

    def forward(self, x):
        return x.permute(self.dims)

# 两个维度交换
class Transpose(nn.Module):
    def __init__(self):  # 默认交换 dim0 和 dim1
        super().__init__()
        self.dim0 = 0
        self.dim1 = 1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)
#endregion


#region 各网络层定义
layer_config = {
#region IO
"Input": { 
            # 白灰渐变
            "color_top": (255, 255, 255),
            "color_bottom": (169, 169, 169),
            "introduction": "输入层,接收的外部数据从此进入",
            "type": "IO"
        },

"Output": { 
            # 灰白渐变
            "color_top": (169, 169, 169),
            "color_bottom": (255, 255, 255),
            "introduction": "输出层,网络的结果从此处输出",
            "type": "IO"
        },
#endregion

#region Conv
"Conv1d": { 
            # 橙色渐变
            "color_top": (255, 165, 0),
            "color_bottom": (255, 65, 0),
            "nn" : nn.Conv1d,
            "default" : (3, 3, 1),
            "introduction": "Pytorch中的一维卷积层 Conv1d",
            "type": "Conv"
        },

"Conv2d": { 
            # 橙色渐变
            "color_top": (255, 165, 0),
            "color_bottom": (255, 65, 0),
            "nn" : nn.Conv2d,
            "default" : (3, 3, 1),
            "introduction": "Pytorch中的二维卷积层 Conv2d",
            "type": "Conv"
        },

"Conv3d": { 
            # 橙色渐变
            "color_top": (255, 165, 0),
            "color_bottom": (255, 65, 0),
            "nn" : nn.Conv3d,
            "default" : (3, 3, 1),
            "introduction": "Pytorch中的三维卷积层 Conv3d",
            "type": "Conv"
        },

"ConvT1d": { 
            # 橙色渐变
            "color_top": (255, 165, 0),
            "color_bottom": (255, 65, 0),
            "nn" : nn.ConvTranspose1d,
            "default" : (3, 3, 1),
            "introduction": "Pytorch中的一维转置卷积层 ConvTranspose1d",
            "type": "Conv"
        },

"ConvT2d": { 
            # 橙色渐变
            "color_top": (255, 165, 0),
            "color_bottom": (255, 65, 0),
            "nn" : nn.ConvTranspose2d,
            "default" : (3, 3, 1),
            "introduction": "Pytorch中的二维转置卷积层 ConvTranspose2d",
            "type": "Conv"
        },

"ConvT3d": { 
            # 橙色渐变
            "color_top": (255, 165, 0),
            "color_bottom": (255, 65, 0),
            "nn" : nn.ConvTranspose3d,
            "default" : (3, 3, 1),
            "introduction": "Pytorch中的三维转置卷积层 ConvTranspose3d",
            "type": "Conv"
        },
#endregion

#region Normalization
"BN1d": {
            # 紫色渐变
            "color_top": (186, 85, 211),
            "color_bottom": (138, 43, 226),
            "nn": nn.BatchNorm1d,
            "default": (64,),
            "introduction": "Pytorch中的一维批归一化层 BatchNorm1d",
            "type": "Normalization"
        },

"BN2d": {
            # 紫色渐变
            "color_top": (186, 85, 211),
            "color_bottom": (138, 43, 226),
            "nn": nn.BatchNorm2d,
            "default": (64,),
            "introduction": "Pytorch中的二维批归一化层 BatchNorm2d",
            "type": "Normalization"
        },

"BN3d": {
            # 紫色渐变
            "color_top": (186, 85, 211),
            "color_bottom": (138, 43, 226),
            "nn": nn.BatchNorm3d,
            "default": (64,),
            "introduction": "Pytorch中的三维批归一化层 BatchNorm3d",
            "type": "Normalization"
        },

  "LN": {
            # 紫色渐变
            "color_top": (186, 85, 211),
            "color_bottom": (138, 43, 226),
            "nn": nn.LayerNorm,
            "default": (512,),
            "introduction": "Pytorch中的层归一化层 LayerNorm",
            "type": "Normalization"
        },

"IN1d": {
            # 紫色渐变
            "color_top": (186, 85, 211),
            "color_bottom": (138, 43, 226),
            "nn": nn.InstanceNorm1d,
            "default": (64,),
            "introduction": "Pytorch中的一维实例归一化层 InstanceNorm1d",
            "type": "Normalization"
        },

"IN2d": {
            # 紫色渐变
            "color_top": (186, 85, 211),
            "color_bottom": (138, 43, 226),
            "nn": nn.InstanceNorm2d,
            "default": (64,),
            "introduction": "Pytorch中的二维实例归一化层 InstanceNorm2d",
            "type": "Normalization"
        },

"IN3d": {
            # 紫色渐变
            "color_top": (186, 85, 211),
            "color_bottom": (138, 43, 226),
            "nn": nn.InstanceNorm3d,
            "default": (64,),
            "introduction": "Pytorch中的三维实例归一化层 InstanceNorm3d",
            "type": "Normalization"
        },

  "GN": {
            # 紫色渐变
            "color_top": (186, 85, 211),
            "color_bottom": (138, 43, 226),
            "nn": nn.GroupNorm,
            "default": (32, 64),
            "introduction": "Pytorch中的分组归一化层 GroupNorm",
            "type": "Normalization"
        },
#endregion

#region Activation
"ReLU": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.ReLU,
            "default": (),
            "introduction": "Pytorch中的激活函数 ReLU ",
            "type": "Activation"
        },

"LeakyReLU": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.LeakyReLU,
            "default": (),
            "introduction": "Pytorch中的激活函数 Leaky ReLU",
            "type": "Activation"
        },

"ReLU6": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.ReLU6,
            "default": (),
            "introduction": "Pytorch中的激活函数 ReLU6 ,限制最大值为6",
            "type": "Activation"
        },

"Sigmoid": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.Sigmoid,
            "default": (),
            "introduction": "Pytorch中的激活函数 Sigmoid",
            "type": "Activation"
        },

"Tanh": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.Tanh,
            "default": (),
            "introduction": "Pytorch中的激活函数 Tanh",
            "type": "Activation"
        },

"Softmax": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.Softmax,
            "default": (-1,),
            "introduction": "Pytorch中的激活函数 Softmax ,通常用于分类任务的最后一层",
            "type": "Activation"
        },

"Softplus": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.Softplus,
            "default": (),
            "introduction": "Pytorch中的激活函数 Softplus",
            "type": "Activation"
        },

"Softsign": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.Softsign,
            "default": (),
            "introduction": "Pytorch中的激活函数 Softsign",
            "type": "Activation"
        },

 "ELU": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.ELU,
            "default": (),
            "introduction": "Pytorch中的激活函数 ELU ,指数线性单元",
            "type": "Activation"
        },

"SELU": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.SELU,
            "default": (),
            "introduction": "Pytorch中的激活函数 SELU ,自带归一化效果",
            "type": "Activation"
        },

"GELU": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.GELU,
            "default": (),
            "introduction": "Pytorch中的激活函数 GELU ,常用于 Transformer 结构",
            "type": "Activation"
        },

"Hardtanh": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.Hardtanh,
            "default": (),
            "introduction": "Pytorch中的激活函数 Hardtanh ,硬Tanh函数",
            "type": "Activation"
        },

"Hardshrink": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.Hardshrink,
            "default": (),
            "introduction": "Pytorch中的激活函数 Hardshrink",
            "type": "Activation"
        },

"LogSigmoid": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.LogSigmoid,
            "default": (),
            "introduction": "Pytorch中的激活函数 LogSigmoid ,对数Sigmoid",
            "type": "Activation"
        },

"PReLU": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.PReLU,
            "default": (),
            "introduction": "Pytorch中的激活函数 PReLU ,参数化ReLU",
            "type": "Activation"
        },

"RReLU": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.RReLU,
            "default": (),
            "introduction": "Pytorch中的激活函数 RReLU ,随机Leaky ReLU",
            "type": "Activation"
        },

"Softshrink": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.Softshrink,
            "default": (),
            "introduction": "Pytorch中的激活函数 Softshrink",
            "type": "Activation"
        },

"Threshold": { 
            # 黄色渐变
            "color_top": (255, 255, 0),
            "color_bottom": (255, 255, 100),
            "nn": nn.Threshold,
            "default": (0.1, 0.2),
            "introduction": "Pytorch中的激活函数 Threshold ,设定阈值",
            "type": "Activation"
        },
#endregion

#region Pooling
"MP1d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.MaxPool1d,
            "default": (2,2),
            "introduction": "Pytorch中的一维最大池化层 MaxPool1d",
            "type": "Pooling"
        },

"MP2d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.MaxPool2d,
            "default": (2,2),
            "introduction": "Pytorch中的二维最大池化层 MaxPool2d",
            "type": "Pooling"
        },

"MP3d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.MaxPool3d,
            "default": (2,2),
            "introduction": "Pytorch中的三维最大池化层 MaxPool3d",
            "type": "Pooling"
        },

"AP1d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.AvgPool1d,
            "default": (2,2),
            "introduction": "Pytorch中的一维平均池化层 AvgPool1d",
            "type": "Pooling"
        },

"AP2d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.AvgPool2d,
            "default": (2,2),
            "introduction": "Pytorch中的二维平均池化层 AvgPool2d",
            "type": "Pooling"
        },

"AP3d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.AvgPool3d,
            "default": (2,2),
            "introduction": "Pytorch中的三维平均池化层 AvgPool3d",
            "type": "Pooling"
        },

"AMP1d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.AdaptiveMaxPool1d,
            "default": (1,),
            "introduction": "Pytorch中的一维自适应最大池化层 AdaptiveMaxPool1d",
            "type": "Pooling"
        },

"AMP2d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.AdaptiveMaxPool2d,
            "default": (1,),
            "introduction": "Pytorch中的二维自适应最大池化层 AdaptiveMaxPool2d",
            "type": "Pooling"
        },

"AMP3d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.AdaptiveMaxPool3d,
            "default": (1,),
            "introduction": "Pytorch中的三维自适应最大池化层 AdaptiveMaxPool3d",
            "type": "Pooling"
        },

"AAP1d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.AdaptiveAvgPool1d,
            "default": (1,),
            "introduction": "Pytorch中的一维自适应平均池化层 AdaptiveAvgPool1d",
            "type": "Pooling"
        },

"AAP2d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.AdaptiveAvgPool2d,
            "default": (1,),
            "introduction": "Pytorch中的二维自适应平均池化层 AdaptiveAvgPool2d",
            "type": "Pooling"
        },

"AAP3d": {
            # 绿色渐变
            "color_top": (144, 238, 144),
            "color_bottom": (34, 139, 34),
            "nn": nn.AdaptiveAvgPool3d,
            "default": (1,),
            "introduction": "Pytorch中的三维自适应平均池化层 AdaptiveAvgPool3d",
            "type": "Pooling"
        },
#endregion

#region Linear
"Linear": {
            # 蓝色渐变
            "color_top": (70, 130, 180),
            "color_bottom": (30, 144, 255),
            "nn": nn.Linear,
            "default": (128, 64),
            "introduction": "Pytorch中的全连接层 Linear ,执行线性变换 y = Wx + b",
            "type": "Linear"
        },

"Bilinear": {
            # 蓝色渐变
            "color_top": (70, 130, 180),
            "color_bottom": (30, 144, 255),
            "nn": nn.Bilinear,
            "default": (128, 64, 32),
            "introduction": "Pytorch中的双线性层 Bilinear ,计算双线性变换 y = x₁ᵀ W x₂ + b",
            "type": "Linear"
        },
#endregion

#region Dropout
"Dropout": {
            # 青色渐变
            "color_top": (0, 206, 209),
            "color_bottom": (0, 139, 139),
            "nn": nn.Dropout,
            "default": (),
            "introduction": "Pytorch中的 Dropout 层",
            "type": "Dropout"
        },

"Dropout2d": {
            # 青色渐变
            "color_top": (0, 206, 209),
            "color_bottom": (0, 139, 139),
            "nn": nn.Dropout2d,
            "default": (),
            "introduction": "Pytorch中的 Dropout2d 层",
            "type": "Dropout"
        },

"Dropout3d": {
            # 青色渐变
            "color_top": (0, 206, 209),
            "color_bottom": (0, 139, 139),
            "nn": nn.Dropout3d,
            "default": (),
            "introduction": "Pytorch中的 Dropout3d 层",
            "type": "Dropout"
        },

"A-Dropout": {
            # 青色渐变
            "color_top": (0, 206, 209),
            "color_bottom": (0, 139, 139),
            "nn": nn.AlphaDropout,
            "default": (),
            "introduction": "Pytorch中的 AlphaDropout 层,适用于 SELU 激活函数的归一化保持",
            "type": "Dropout"
        },
#endregion

#region Recurrent
"RNN": {
            # 浅蓝色渐变
            "color_top": (135, 206, 250),  
            "color_bottom": (70, 130, 180),  
            "nn": nn.RNN,
            "default": (128, 64),
            "introduction": "Pytorch中的循环神经网络层 RNN ,适用于序列数据处理",
            "type": "Recurrent"
        },

"GRU": {
            # 浅蓝色渐变
            "color_top": (135, 206, 250),  
            "color_bottom": (70, 130, 180),  
            "nn": nn.GRU,
            "default": (128, 64),
            "introduction": "Pytorch中的门控循环单元层 GRU ,改进的 RNN 变体",
            "type": "Recurrent"
        },

"LSTM": {
            # 浅蓝色渐变
            "color_top": (135, 206, 250),  
            "color_bottom": (70, 130, 180),  
            "nn": nn.LSTM,
            "default": (128, 64),
            "introduction": "Pytorch中的长短时记忆网络层 LSTM ,适用于长序列建模",
            "type": "Recurrent"
        },

"RNNCell": {
            # 浅蓝色渐变
            "color_top": (135, 206, 250),  
            "color_bottom": (70, 130, 180),  
            "nn": nn.RNNCell,
            "default": (128, 64),
            "introduction": "Pytorch中的 RNNCell ,单步 RNN 计算单元",
            "type": "Recurrent"
        },

"GRUCell": {
            # 浅蓝色渐变
            "color_top": (135, 206, 250),  
            "color_bottom": (70, 130, 180),  
            "nn": nn.GRUCell,
            "default": (128, 64),
            "introduction": "Pytorch中的 GRUCell ,单步 GRU 计算单元",
            "type": "Recurrent"
        },

"LSTMCell": {
            # 浅蓝色渐变
            "color_top": (135, 206, 250),  
            "color_bottom": (70, 130, 180),  
            "nn": nn.LSTMCell,
            "default": (128, 64),
            "introduction": "Pytorch中的 LSTMCell ,单步 LSTM 计算单元",
            "type": "Recurrent"
        },
#endregion

#region Transformer
"Trans": {
            # 深紫色渐变
            "color_top": (75, 0, 130),
            "color_bottom": (138, 43, 226),
            "nn": nn.Transformer,
            "default": (),
            "introduction": "Pytorch中的层 Transformer ,包含完整的编码器-解码器结构",
            "type": "Transformer"
        },

"TransEnc": {
            # 深紫色渐变
            "color_top": (75, 0, 130),
            "color_bottom": (138, 43, 226),
            "nn": nn.TransformerEncoder,
            "default": (nn.TransformerEncoderLayer(d_model=128, nhead=8), 6),
            "introduction": "Pytorch中的层 TransformerEncoder ,包含多个编码器层",
            "type": "Transformer"
        },

"TransDec": {
            # 深紫色渐变
            "color_top": (75, 0, 130),
            "color_bottom": (138, 43, 226),
            "nn": nn.TransformerDecoder,
            "default": (nn.TransformerDecoderLayer(d_model=128, nhead=8), 6),
            "introduction": "Pytorch中的层 TransformerDecoder ,包含多个解码器层",
            "type": "Transformer"
        },

"TransEncL": {
            # 深紫色渐变
            "color_top": (75, 0, 130),
            "color_bottom": (138, 43, 226),
            "nn": nn.TransformerEncoderLayer,
            "default": (128, 8),
            "introduction": "Pytorch中的层 TransformerEncoderLayer ,单个编码器层",
            "type": "Transformer"
        },

"TransDecL": {
            # 深紫色渐变
            "color_top": (75, 0, 130),
            "color_bottom": (138, 43, 226),
            "nn": nn.TransformerDecoderLayer,
            "default": (128, 8),
            "introduction": "Pytorch中的层 TransformerDecoderLayer ,单个解码器层",
            "type": "Transformer"
        },

"MHA": {
            # 深紫色渐变
            "color_top": (75, 0, 130),
            "color_bottom": (138, 43, 226),
            "nn": nn.MultiheadAttention,
            "default": (128, 8),
            "introduction": "Pytorch中的层 MultiheadAttention ,多头自注意力机制",
            "type": "Transformer"
        },
#endregion

#region TensorOps
"Flatten": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": nn.Flatten,
    "default": (),
    "introduction": "将多维张量展平",
    "type": "TensorOps"
},

"Unflatten": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": nn.Unflatten,
    "default": (1, (64, 64)),
    "introduction": "将张量重塑为更多维",
    "type": "TensorOps"
},

"Add": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Add,
    "default": (),
    "introduction": "逐元素相加",
    "type": "TensorOps"
},

"Mul": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Mul,
    "default": (),
    "introduction": "逐元素相乘",
    "type": "TensorOps"
},

"Maximum": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Maximum,
    "default": (),
    "introduction": "逐元素求最大值",
    "type": "TensorOps"
},

"Minimum": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Minimum,
    "default": (),
    "introduction": "逐元素求最小值",
    "type": "TensorOps"
},

"Mean": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Mean,
    "default": (),
    "introduction": "指定维度求均值",
    "type": "TensorOps"
},

"Sum": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Sum,
    "default": (),
    "introduction": "指定维度求和",
    "type": "TensorOps"
},

"Cat": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Cat,
    "default": (),
    "introduction": "沿指定维度拼接张量",
    "type": "TensorOps"
},

"Stack": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Stack,
    "default": (),
    "introduction": "沿指定维度堆叠张量",
    "type": "TensorOps"
},

"Split": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Split,
    "default": (),
    "introduction": "将张量拆分为多个子张量",
    "type": "TensorOps"
},

"View": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": View,
    "default": (),
    "introduction": "改变张量形状",
    "type": "TensorOps"
},

"Permute": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Permute,
    "default": (),
    "introduction": "交换张量的多个维度顺序",
    "type": "TensorOps"
},

"Transpose": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Transpose,
    "default": (),
    "introduction": "交换张量的两个维度",
    "type": "TensorOps"
},

"Repeat": {
    "color_top": (105, 105, 105),
    "color_bottom": (169, 169, 169),
    "nn": Repeat,
    "default": (),
    "introduction": "沿指定维度重复张量",
    "type": "TensorOps"
},
#endregion

}
#endregion


# LaneSentry

#### 一个基于深度学习的消防通道车辆车牌检测工具

🚗 **车辆与车牌检测** | 🛠️ **基于YOLOv11与ONNX构建** | 📊 **可视化看板**

## 📸 演示效果

![演示](example.gif)

## 🔍 项目概述

LaneSentry 是一个AI驱动的交通监控解决方案，具备以下功能：

- **实时车辆检测**：基于YOLOv11的车辆识别
- **车牌二次识别**：通过二级YOLO模型精确定位车牌
- **微信堵塞通知**：根据检测结果发送通知
- **边缘计算优化**：使用ONNX Runtime实现高效推理

## 🚀 应用场景

适用于消防通道车辆检测，避免堵塞引起的紧急情况。

## 🛠️ 安装指南

```CLI
git clone https://github.com/Youshuizhikon/LaneSentry.git
cd LaneSentry
uv sync  # 最好Python 3.12+
python run mian.py
```
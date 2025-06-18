# 一个通俗的 Swin Transformer  demo  使用 MSRA 的源码 

## 简介
- 微软亚洲研究院(MSRA)的工作
- 参考视频：[Swin Transformer详解](https://www.bilibili.com/video/BV13L4y1475U)

## Swin Transformer vs ViT
Swin Transformer 在多个领域都取得了优异的成绩，包括：
- 图像分类 (Image Classification)
- 图像分割 (Segmentation)
- 目标检测 (Object Detection)

### 主要优势 参考 Figure 1 模型架构对比图
1. **Token分辨率**： 
   - ViT：使用单一低分辨率token (16)
   - Swin Transformer：token分辨率是动态变化的

2. **计算复杂度**：
   - Swin Transformer取消了全局视野
   - 窗口大小固定，注意力计算复杂度恒定
   - 在局部窗口内进行self-attention，避免全局注意力计算

3. **多尺度特征**：
   - 使用patch merging操作
   - 模拟传统CNN中的下采样操作
   - 通过1×1卷积进行通道降维

## 核心机制

### Shift Window机制 参考 Figure 2: Shift Window原理图
- 将窗口向左下移动两个patch
- 实现跨窗口连接(cross window connection)
- 灰色区域代表patch，红色区域代表self-attention窗口

### 前向传播过程 参考 Figure 3: 模型前向过程
以224×224×3的输入图像为例：
1. Patch操作(4×4) → 56×56×(4×4×3)
2. Linear Embedding → 56×56×96
3. 序列长度达到3136
4. 在49个patch内进行self-attention

### 注意力计算
- 首先进行W-MSA (Window Multi-head Self-Attention)
- 然后进行SW-MSA (Shifted Window Multi-head Self-Attention)
- 使用Mask掩码处理移动窗口后的不完整注意力计算


- 
- 
### Figure 4: Mask掩码实现示意图  讲解省略


### demo run  swin-transformer\test_swin.py

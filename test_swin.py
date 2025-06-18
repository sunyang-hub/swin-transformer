import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'Swin_Transformer_main', 'Swin-Transformer-main'))

from models.swin_transformer import SwinTransformer

class TrainSet(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

def print_model_structure(model):
    """打印模型结构"""
    print("\n模型结构:")
    print(model)
    
    # 打印每一层的参数数量
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} 参数")
            total_params += param.numel()
    print(f"\n总参数量: {total_params:,}")

def main():
    # 1. 准备数据
    # 创建模拟的3D图像数据 (batch_size, channels, depth, height, width)
    X_tensor = torch.randn((32, 3,  224, 224))  # 3通道,224x224分辨率
    Y_tensor = torch.zeros((32, 1000))  # 1000分类任务
    
    mydataset = TrainSet(X_tensor, Y_tensor)
    train_loader = DataLoader(mydataset, batch_size=2, shuffle=True)

    # 2. 初始化Swin Transformer模型
    net = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )
    
    # 打印模型结构
    print_model_structure(net)

    # 3. 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

    # 4. 训练循环
    print("\n开始训练...")
    for epoch in range(2):  # 为了演示,只训练2个epoch
        for i, (X, y) in enumerate(train_loader):
            # 前向传播
            pred = net(X)
            
            # 计算损失
            loss = loss_fn(pred, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 5 == 0:  # 每5个batch打印一次
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.4f}')
                
                # 打印一些中间层的输出形状
                if i == 0:
                    print("\n前向传播过程中的特征图形状:")
                    print(f"输入形状: {X.shape}")
                    print(f"输出形状: {pred.shape}")

if __name__ == '__main__':
    main() 
# config.py
"""
配置文件：集中管理所有超参数、路径和设备设置。
"""

import os
import torch

class Config:
    """全局配置类，所有参数通过类属性访问"""

    # ==================== 数据参数 ====================
    # 真实MIMIC-III-Ext-PPG数据集的存放路径（当use_synthetic=False时需准备好）
    data_path = "./data/mimic_iii_ext_ppg"

    # 是否使用模拟数据（True: 不依赖真实数据，快速测试代码逻辑；False: 加载真实数据）
    use_synthetic = True

    # 模拟数据的样本数量（仅use_synthetic=True时生效）
    synthetic_samples = 100

    # 每个PPG窗口的长度（点数），30秒 * 125Hz = 3750点
    window_length = 3750

    # 信号采样率 (Hz)
    sampling_rate = 125

    # 信号质量标签二值化阈值：MIMIC-III-Ext-PPG中的SQI整数码（0=差,1=尚可,2=好）
    # 此处设定 >=1 视为高质量（标签=1），<1视为低质量（标签=0）
    sqi_threshold = 1

    # ==================== 模型架构参数 ====================
    # 输入通道数：红光PPG + 红外PPG
    input_channels = 2

    # 隐藏层维度（特征图通道数），也是条件嵌入向量的维度
    hidden_dim = 64

    # 交叉注意力头的数量
    num_heads = 8

    # Dropout比例，用于正则化防止过拟合
    dropout = 0.1

    # ==================== 训练参数 ====================
    batch_size = 64  # 批大小（显存不足可减小）
    epochs = 50  # 训练轮数
    lr = 1e-3  # 初始学习率
    weight_decay = 1e-5  # AdamW优化器的权重衰减系数（L2正则化）

    # 多任务损失权重：主任务（血压预测）权重 = lambda_task，
    # 辅助任务（SQI预测）权重 = 1 - lambda_task
    lambda_task = 0.7

    # ==================== 硬件与保存 ====================
    # 自动选择cuda（GPU）或cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型保存目录
    save_dir = "./checkpoints"
    # 日志目录（训练曲线图等）
    log_dir = "./logs"

    # 自动创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
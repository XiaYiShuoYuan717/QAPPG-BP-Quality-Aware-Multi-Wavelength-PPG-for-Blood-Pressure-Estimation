# dataset.py
"""
数据集加载模块：
- 支持模拟数据生成（用于快速验证代码）
- 提供真实MIMIC-III-Ext-PPG数据加载的骨架（需用户根据实际文件结构实现）
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class PPG_BP_Dataset(Dataset):
    """PPG血压预测数据集"""

    def __init__(self, config, mode='train'):
        """
        参数:
            config: Config对象，包含所有参数
            mode: 仅用于标识（实际未使用，但可扩展用于训练/测试区分）
        """
        self.config = config
        self.mode = mode

        if config.use_synthetic:
            # 使用模拟数据（快速测试）
            self._generate_synthetic_data()
        else:
            # 使用真实MIMIC-III-Ext-PPG数据集（需自行实现）
            self._load_real_data()

    def _generate_synthetic_data(self):
        """
        生成模拟的PPG信号和对应的血压标签、质量标签。
        信号模型：基频为1.25Hz（模拟75bpm心率），叠加二次谐波，
        并添加不同程度的噪声来模拟低/中/高质量。
        """
        num_samples = self.config.synthetic_samples
        T = self.config.window_length  # 3750点

        # 时间轴（0 到 2π 长度，用于生成周期信号）
        t = np.linspace(0, 2 * np.pi, T)

        self.signals = []  # 存储每个样本的(2, T)数组
        self.bps = []  # 存储[SBP, DBP]
        self.sqi_labels = []  # 存储0或1的质量标签（1表示高质量）

        for i in range(num_samples):
            # 基础频率 1.25Hz (75 beats per minute)
            freq = 1.25
            # 生成红光信号：基频正弦 + 二次谐波
            red = 0.5 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(4 * np.pi * freq * t)
            # 生成红外信号：相位偏移0.2弧度，幅值稍不同
            ir = 0.5 * np.sin(2 * np.pi * freq * t + 0.2) + 0.15 * np.sin(4 * np.pi * freq * t)

            # 随机决定信号质量等级 (0:低质量, 1:高质量)
            # 这里为了有监督学习，模拟30%低质量，70%高质量
            if random.random() < 0.3:
                noise_level = 0.3  # 高噪声
                sqi = 0
            else:
                noise_level = 0.02  # 低噪声
                sqi = 1
            # 添加高斯白噪声
            red += noise_level * np.random.randn(T)
            ir += noise_level * np.random.randn(T)

            # 标准化：零均值，单位方差（有助于模型训练）
            red = (red - red.mean()) / (red.std() + 1e-6)
            ir = (ir - ir.mean()) / (ir.std() + 1e-6)

            # 合成双通道信号
            signal = np.stack([red, ir], axis=0)  # (2, T)

            # 模拟血压：与信号的平均幅值线性相关（模拟生理逻辑）
            # 加入随机噪声模拟个体差异
            sbp = 110 + 20 * (red.mean() + ir.mean()) + 5 * np.random.randn()
            dbp = 70 + 15 * (red.mean() + ir.mean()) + 3 * np.random.randn()
            # 限制血压在生理合理范围
            sbp = np.clip(sbp, 80, 180)
            dbp = np.clip(dbp, 50, 120)

            self.signals.append(signal.astype(np.float32))
            self.bps.append([sbp, dbp])
            self.sqi_labels.append(sqi)

        # 转换为numpy数组以便索引
        self.signals = np.array(self.signals)  # (N, 2, T)
        self.bps = np.array(self.bps)  # (N, 2)
        self.sqi_labels = np.array(self.sqi_labels)  # (N,)
        print(f"Generated {num_samples} synthetic samples.")

    def _load_real_data(self):
        """
        真实数据加载函数（需根据MIMIC-III-Ext-PPG的实际文件结构实现）
        建议步骤：
        1. 使用wfdb库读取记录文件（如*.dat, *.hea）
        2. 提取PPG信号（红光和红外通道）、ABP信号（参考血压）
        3. 将ABP信号转换为每个窗口的SBP/DBP标签（可采用滑动窗口取最大值/最小值）
        4. 利用数据集自带的SQI字段生成质量标签
        5. 填充self.signals, self.bps, self.sqi_labels
        """
        import wfdb
        # 示例：假设数据目录下包含记录列表文件RECORDS
        # 实际使用时需要遍历所有记录并生成窗口
        raise NotImplementedError("Real data loading not implemented. Set use_synthetic=True for testing.")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        """
        返回:
            signal: torch.Tensor (2, T)  双波长信号
            bp:     torch.Tensor (2,)    [SBP, DBP]
            sqi:    torch.Tensor (1,)    质量标签（0或1）
        """
        signal = torch.from_numpy(self.signals[idx])  # (2, T)
        bp = torch.from_numpy(self.bps[idx])  # (2,)
        sqi = torch.tensor(self.sqi_labels[idx], dtype=torch.float32).unsqueeze(0)  # (1,)
        return signal, bp, sqi


def get_dataloaders(config):
    """构建训练和验证的DataLoader，自动按8:2分割数据集"""
    dataset = PPG_BP_Dataset(config, mode='train')
    total = len(dataset)
    split = int(0.8 * total)
    # 随机分割（固定种子保证可复现）
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [split, total - split],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
# models.py
"""
定义所有神经网络模块：
- SEBlock: 通道注意力机制
- SQIEstimator: 信号质量评估模块
- QualityAwareFeatureExtractor (QAFE): 质量感知动态特征提取
- CrossAttentionFusion: 双波长交叉注意力融合
- MultiTaskBPModel: 完整的多任务血压预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ========================= 辅助模块：SEBlock =========================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 模块（通道注意力）
    作用：自动学习每个特征通道的重要性权重，增强有用通道，抑制无用通道。
    """

    def __init__(self, channels, reduction=16):
        """
        参数:
            channels: 输入特征图的通道数
            reduction: 降维比例，用于减少参数量。默认16
        """
        super().__init__()
        # 两个全连接层组成门控机制
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),  # 非线性激活
            nn.Linear(channels // reduction, channels, bias=False),  # 恢复维度
            nn.Sigmoid()  # 输出0~1之间的权重
        )

    def forward(self, x):
        """
        参数:
            x: 输入特征图，形状 (batch_size, channels, time_steps)
        返回:
            加权后的特征图，形状与输入相同
        """
        b, c, t = x.size()
        # Squeeze: 全局平均池化，将每个通道的时间维度压缩为一个标量
        y = x.mean(dim=-1)  # (b, c)
        # Excitation: 学习通道权重
        y = self.fc(y).view(b, c, 1)  # (b, c, 1)
        # Scale: 将权重广播到时间维度，与原始特征相乘
        return x * y.expand_as(x)


# ========================= 模块1：SQI估计器 =========================
class SQIEstimator(nn.Module):
    """
    双波长信号质量估计模块
    输入: (batch, 2, time_steps)   – 红光和红外PPG
    输出:
        - sqi (batch, 1)          : 预测的信号质量分数（0~1）
        - condition_embed (batch, hidden_dim) : 条件嵌入向量，供后续模块使用
    """

    def __init__(self, input_dim=2, hidden_dim=64, output_dim=64, kernel_size=15, stride=2):
        """
        参数:
            input_dim: 输入通道数（2）
            hidden_dim: 隐藏层通道数
            output_dim: 条件嵌入向量的维度
            kernel_size: 卷积核大小
            stride: 卷积步长（用于下采样，减少序列长度）
        """
        super().__init__()
        # 第一层卷积：初步特征提取，同时降低时间分辨率（stride=2）
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        # SE通道注意力模块，增强重要特征
        self.se = SEBlock(hidden_dim)
        # 第二层卷积：进一步提取深层特征，保持时间长度（stride=1）
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        # 全局平均池化：将时间维度压缩为1
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 分支1：SQI预测头（二分类，输出0~1的连续值）
        self.sqi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 确保输出在[0,1]区间
        )
        # 分支2：条件嵌入头，输出一个向量供QAFE模块使用
        self.embed_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        参数:
            x: (batch, 2, time_steps) 原始双波长PPG信号
        返回:
            sqi: (batch, 1)           预测的信号质量分数
            cond_embed: (batch, output_dim)  条件嵌入
        """
        # 步骤1: 卷积特征提取
        out = self.conv1(x)  # (b, hidden_dim, T/stride)
        out = self.se(out)  # 通道注意力加权
        out = self.conv2(out)  # (b, hidden_dim, T/stride)

        # 步骤2: 全局池化得到每个通道的全局描述符
        pooled = self.gap(out).squeeze(-1)  # (b, hidden_dim)

        # 步骤3: 分别计算SQI和条件嵌入
        sqi = self.sqi_head(pooled)  # (b, 1)
        cond_embed = self.embed_head(pooled)  # (b, output_dim)

        return sqi, cond_embed


# ========================= 模块2：质量感知动态特征提取器 (QAFE) =========================
class QualityAwareFeatureExtractor(nn.Module):
    """
    根据条件嵌入（包含信号质量信息）动态融合三条不同感受野的编码路径。
    三条路径：
        - 高质量路径：小卷积核（感受野小），保留精细波形细节
        - 中质量路径：标准卷积核（感受野中等）
        - 低质量路径：空洞卷积（感受野大），捕获长程依赖，对噪声鲁棒
    """

    def __init__(self, input_dim=2, hidden_dim=64, cond_dim=64, time_steps=3750):
        """
        参数:
            input_dim: 输入通道数（2）
            hidden_dim: 所有编码器输出特征的通道数
            cond_dim: 条件嵌入的维度（必须与SQIEstimator的输出维度一致）
            time_steps: 输入时间长度（未使用，但保留用于可能的形状自适应）
        """
        super().__init__()
        # ----- 高质量编码器（小感受野）-----
        # 使用kernel_size=5，步长=1，padding=2保持长度不变
        self.high_enc = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # ----- 中质量编码器（中等感受野）-----
        self.mid_enc = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # ----- 低质量编码器（大感受野，空洞卷积）-----
        # dilation=2表示间隔采样，感受野扩大为 kernel_size + (dilation-1)*(kernel_size-1)
        # padding需要调整为 dilation * (kernel_size-1)/2 以保证输出长度不变
        self.low_enc = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=9, padding=8, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=8, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # ----- 动态权重生成网络（超网络）-----
        # 输入：条件嵌入 (cond_dim)
        # 输出：3个权重（softmax归一化），分别对应高、中、低质量路径的融合系数
        self.gate_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 输出3个logits
            nn.Softmax(dim=-1)  # 转换为和为1的概率分布
        )

        # 投影层：将所有编码器的输出对齐到相同维度（实际上三条路径输出维度已经相同，但保留1x1卷积用于特征变换）
        self.proj = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x, cond_embed):
        """
        参数:
            x: (batch, 2, time_steps) 原始双波长信号
            cond_embed: (batch, cond_dim) 由SQIEstimator产生的条件嵌入
        返回:
            fused: (batch, hidden_dim, time_steps) 动态融合后的特征图（时间维度与输入相同）
            weights: (batch, 3) 每个样本对应的三条路径的权重（可用于分析或可视化）
        """
        # 1. 分别通过三个编码器
        f_high = self.high_enc(x)  # (b, hidden_dim, T)
        f_mid = self.mid_enc(x)  # (b, hidden_dim, T)
        f_low = self.low_enc(x)  # (b, hidden_dim, T)

        # 2. 根据条件嵌入计算动态权重
        weights = self.gate_net(cond_embed)  # (b, 3)
        # 拆分为三个权重向量，并增加维度用于广播到空间维度
        w_high = weights[:, 0:1].unsqueeze(-1)  # (b, 1, 1)
        w_mid = weights[:, 1:2].unsqueeze(-1)  # (b, 1, 1)
        w_low = weights[:, 2:3].unsqueeze(-1)  # (b, 1, 1)

        # 3. 加权融合
        fused = w_high * f_high + w_mid * f_mid + w_low * f_low  # (b, hidden_dim, T)
        # 4. 投影变换（可选）
        fused = self.proj(fused)
        return fused, weights


# ========================= 模块3：双波长交叉注意力融合 =========================
class CrossAttentionFusion(nn.Module):
    """
    实现红光与红外特征之间的双向交叉注意力。
    核心思想：让红光特征查询红外特征的关键位置，反之亦然，从而建立双波长之间的时间对齐。
    """

    def __init__(self, dim=64, num_heads=8):
        """
        参数:
            dim: 输入特征的通道数（每个时间点的特征维度）
            num_heads: 多头注意力的头数
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 确保dim可以被num_heads整除
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5  # 缩放因子，避免点积过大

        # 定义红光->红外注意力中的Q, K, V线性变换
        self.q_red = nn.Linear(dim, dim)
        self.k_ir = nn.Linear(dim, dim)
        self.v_ir = nn.Linear(dim, dim)

        # 定义红外->红光注意力中的Q, K, V线性变换
        self.q_ir = nn.Linear(dim, dim)
        self.k_red = nn.Linear(dim, dim)
        self.v_red = nn.Linear(dim, dim)

        # 输出投影层
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, feat_red, feat_ir):
        """
        参数:
            feat_red: 红光特征，形状 (batch, dim, time_steps)
            feat_ir:  红外特征，形状 (batch, dim, time_steps)
        返回:
            fused: 融合后的特征，形状 (batch, dim, time_steps)
        """
        B, D, T = feat_red.shape
        # 将形状转为 (batch, time_steps, dim) 以方便进行序列注意力计算
        f_red = feat_red.permute(0, 2, 1)  # (B, T, D)
        f_ir = feat_ir.permute(0, 2, 1)  # (B, T, D)

        # ---------- 方向1: 红光查询红外 ----------
        # 线性变换并切分为多头
        Q_red = self.q_red(f_red).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nH, T, head_dim)
        K_ir = self.k_ir(f_ir).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nH, T, head_dim)
        V_ir = self.v_ir(f_ir).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nH, T, head_dim)

        # 计算注意力分数 (Q·K^T) / sqrt(d_k)
        attn_red_to_ir = (Q_red @ K_ir.transpose(-2, -1)) * self.scale  # (B, nH, T, T)
        attn_red_to_ir = attn_red_to_ir.softmax(dim=-1)  # 对最后一个维度（键）softmax
        # 加权求和
        out_red_to_ir = (attn_red_to_ir @ V_ir).transpose(1, 2).reshape(B, T, D)  # (B, T, D)

        # ---------- 方向2: 红外查询红光 ----------
        Q_ir = self.q_ir(f_ir).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K_red = self.k_red(f_red).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V_red = self.v_red(f_red).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_ir_to_red = (Q_ir @ K_red.transpose(-2, -1)) * self.scale
        attn_ir_to_red = attn_ir_to_red.softmax(dim=-1)
        out_ir_to_red = (attn_ir_to_red @ V_red).transpose(1, 2).reshape(B, T, D)

        # 融合两个方向的结果（简单平均）
        fused = (out_red_to_ir + out_ir_to_red) / 2  # (B, T, D)
        fused = self.out_proj(fused).permute(0, 2, 1)  # 转回 (B, D, T)
        return fused


# ========================= 完整的多任务模型 =========================
class MultiTaskBPModel(nn.Module):
    """
    整合所有模块：
        - SQI估计器 -> 产生质量分数和条件嵌入
        - 双路特征提取器（这里使用两个轻量CNN分别提取红光/红外特征，简化实现但不失本质）
        - 交叉注意力融合
        - 血压预测头 + 辅助SQI预测头
    注意：原论文设计的QAFE模块理论上应分别应用于红光和红外，为保持代码可读性，
          这里使用两个独立的轻量CNN完成特征提取，也可以替换为两个并行的QAFE实例（需要调整）。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # ---------- SQI估计模块 ----------
        self.sqi_estimator = SQIEstimator(
            input_dim=2,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim  # 条件嵌入维度
        )

        # ---------- 红光和红外各自的特征提取器（简化版本，实际可以用QAFE）----------
        # 设计两个小型CNN网络，分别将单通道PPG映射为 (batch, hidden_dim, time')
        # 注意：这里的时间长度会因卷积步长而缩短，后续交叉注意力会自动处理长度对齐
        self.feat_red = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(32, config.hidden_dim, kernel_size=7, padding=3),
            nn.ReLU()
        )
        self.feat_ir = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(32, config.hidden_dim, kernel_size=7, padding=3),
            nn.ReLU()
        )

        # ---------- 交叉注意力融合 ----------
        self.cross_attn = CrossAttentionFusion(
            dim=config.hidden_dim,
            num_heads=config.num_heads
        )

        # ---------- 血压预测头（主任务）----------
        # 先全局平均池化将时间维度压缩，再通过全连接层输出SBP和DBP
        self.bp_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, hidden_dim, 1)
            nn.Flatten(),  # (B, hidden_dim)
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 2)  # 输出两个值：SBP, DBP
        )

        # ---------- 辅助SQI预测头（辅助任务）----------
        # 同样使用池化+全连接，输出一个0~1的质量分数
        self.aux_sqi_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_weights=False):
        """
        参数:
            x: (batch, 2, time_steps) 输入的双波长PPG信号（通道0=红光，通道1=红外）
            return_weights: 是否额外返回QAFE中的权重（用于分析或可视化）
        返回:
            bp_out: (batch, 2)  预测的SBP, DBP
            sqi_pred: (batch, 1) SQI估计器直接输出的质量分数（可选）
            aux_sqi: (batch, 1)  辅助头预测的质量分数（用于多任务损失）
            [weights]: 仅当return_weights=True时返回
        """
        # 1. 估计信号质量，获得条件嵌入
        sqi_pred, cond_embed = self.sqi_estimator(x)  # sqi_pred: (B,1), cond_embed: (B, D)

        # 2. 分离红光和红外通道
        x_red = x[:, 0:1, :]  # (B, 1, T)
        x_ir = x[:, 1:2, :]  # (B, 1, T)

        # 3. 提取各自的特征
        feat_red = self.feat_red(x_red)  # (B, D, T') ，T' < T
        feat_ir = self.feat_ir(x_ir)  # (B, D, T')

        # 4. 保证两个特征的时间长度一致（正常情况下应当相同，若因填充边界差异可做对齐）
        min_len = min(feat_red.shape[-1], feat_ir.shape[-1])
        if feat_red.shape[-1] != min_len:
            feat_red = feat_red[..., :min_len]
            feat_ir = feat_ir[..., :min_len]

        # 5. 交叉注意力融合
        fused = self.cross_attn(feat_red, feat_ir)  # (B, D, T')

        # 6. 主任务和辅助任务预测
        bp_out = self.bp_head(fused)  # (B, 2)
        aux_sqi = self.aux_sqi_head(fused)  # (B, 1)

        if return_weights:
            # 注意：当前简化版本未使用QAFE，因此weights为None；如果后续替换为QAFE，可在此返回真实权重
            # 这里为了接口一致性，返回一个占位符
            return bp_out, sqi_pred, aux_sqi, None
        return bp_out, sqi_pred, aux_sqi
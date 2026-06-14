# train.py
"""
训练与验证流程：
- 每个epoch执行训练和验证
- 计算多任务损失（血压MSE + SQI二分类交叉熵）
- 记录最佳模型并绘制学习曲线
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

from config import Config
from models import MultiTaskBPModel
from dataset import get_dataloaders


def train_one_epoch(model, loader, optimizer, criterion_bp, criterion_sqi, lambda_task, device):
    """
    单个epoch的训练函数
    返回: 平均总损失、平均血压损失、平均SQI损失
    """
    model.train()
    total_loss = 0.0
    total_bp_loss = 0.0
    total_sqi_loss = 0.0

    # tqdm显示进度条
    for signals, bps, sqis in tqdm(loader, desc="Training"):
        # 将数据移到指定设备（GPU/CPU）
        signals = signals.to(device)  # (B, 2, T)
        bps = bps.to(device)  # (B, 2)
        sqis = sqis.to(device)  # (B, 1)

        optimizer.zero_grad()

        # 前向传播
        bp_pred, sqi_pred, aux_sqi = model(signals)  # bp_pred: (B,2), aux_sqi: (B,1)

        # 计算主任务损失（血压预测）
        loss_bp = criterion_bp(bp_pred, bps)
        # 计算辅助任务损失（SQI预测，使用aux_sqi而非sqi_pred，两者均可，这里用辅助头）
        loss_sqi = criterion_sqi(aux_sqi, sqis)
        # 多任务加权总损失
        loss = lambda_task * loss_bp + (1 - lambda_task) * loss_sqi

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 累计统计
        total_loss += loss.item()
        total_bp_loss += loss_bp.item()
        total_sqi_loss += loss_sqi.item()

    # 返回平均损失
    avg_loss = total_loss / len(loader)
    avg_bp_loss = total_bp_loss / len(loader)
    avg_sqi_loss = total_sqi_loss / len(loader)
    return avg_loss, avg_bp_loss, avg_sqi_loss


def validate(model, loader, criterion_bp, criterion_sqi, lambda_task, device):
    """
    验证函数，返回损失和血压MAE指标
    """
    model.eval()
    total_loss = 0.0
    total_bp_loss = 0.0
    total_sqi_loss = 0.0
    all_bp_pred = []
    all_bp_true = []

    with torch.no_grad():
        for signals, bps, sqis in tqdm(loader, desc="Validating"):
            signals = signals.to(device)
            bps = bps.to(device)
            sqis = sqis.to(device)

            bp_pred, _, aux_sqi = model(signals)
            loss_bp = criterion_bp(bp_pred, bps)
            loss_sqi = criterion_sqi(aux_sqi, sqis)
            loss = lambda_task * loss_bp + (1 - lambda_task) * loss_sqi

            total_loss += loss.item()
            total_bp_loss += loss_bp.item()
            total_sqi_loss += loss_sqi.item()

            # 收集预测值和真值，用于计算MAE
            all_bp_pred.append(bp_pred.cpu().numpy())
            all_bp_true.append(bps.cpu().numpy())

    # 计算整体MAE
    all_bp_pred = np.concatenate(all_bp_pred, axis=0)  # (N, 2)
    all_bp_true = np.concatenate(all_bp_true, axis=0)  # (N, 2)
    mae_sbp = np.mean(np.abs(all_bp_pred[:, 0] - all_bp_true[:, 0]))
    mae_dbp = np.mean(np.abs(all_bp_pred[:, 1] - all_bp_true[:, 1]))

    avg_loss = total_loss / len(loader)
    avg_bp_loss = total_bp_loss / len(loader)
    avg_sqi_loss = total_sqi_loss / len(loader)
    return avg_loss, avg_bp_loss, avg_sqi_loss, mae_sbp, mae_dbp


def train(config):
    """主训练流程"""
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # 1. 加载数据
    train_loader, val_loader = get_dataloaders(config)

    # 2. 初始化模型
    model = MultiTaskBPModel(config).to(device)

    # 3. 定义损失函数
    criterion_bp = nn.MSELoss()  # 血压回归使用均方误差
    criterion_sqi = nn.BCELoss()  # SQI二分类使用二值交叉熵

    # 4. 优化器与学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 5. 训练循环
    best_val_mae = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae_sbp': [],
        'val_mae_dbp': []
    }

    for epoch in range(1, config.epochs + 1):
        # 训练
        train_loss, train_bp_loss, train_sqi_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_bp, criterion_sqi,
            config.lambda_task, device
        )
        # 验证
        val_loss, val_bp_loss, val_sqi_loss, mae_sbp, mae_dbp = validate(
            model, val_loader, criterion_bp, criterion_sqi,
            config.lambda_task, device
        )
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae_sbp'].append(mae_sbp)
        history['val_mae_dbp'].append(mae_dbp)

        # 打印信息
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"SBP MAE: {mae_sbp:.2f} mmHg | DBP MAE: {mae_dbp:.2f} mmHg")

        # 保存最佳模型（按SBP+DBP MAE之和最小）
        current_mae_sum = mae_sbp + mae_dbp
        if current_mae_sum < best_val_mae:
            best_val_mae = current_mae_sum
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pth"))
            print("  -> Best model saved.")

    # 6. 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['val_mae_sbp'], label='SBP MAE')
    plt.plot(history['val_mae_dbp'], label='DBP MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (mmHg)')
    plt.legend()
    plt.title('Validation MAE')
    plt.tight_layout()
    plt.savefig(os.path.join(config.log_dir, 'training_curves.png'))
    plt.show()

    print("Training completed.")
    return model, history
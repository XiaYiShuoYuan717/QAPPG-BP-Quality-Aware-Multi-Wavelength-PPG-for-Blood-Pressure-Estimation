# main.py

from config import Config
from train import train

if __name__ == "__main__":
    # 实例化配置
    config = Config()
    # 开始训练（如果use_synthetic=True，会生成模拟数据并训练）
    model, history = train(config)
    print("All done. Best model saved in checkpoints/best_model.pth")

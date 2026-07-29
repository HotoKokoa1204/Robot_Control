import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import yaml
from agilab.tku.visual_navigation_system.dataset import get_dataloader
from agilab.tku.visual_navigation_system.models import AutoEncoder
from agilab.tku.visual_navigation_system.trainer import Trainer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 確保可正確載入 core 目錄下的模組
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
)


def main():
    # 讀取設定檔
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_config = config["training"]

    # 取得超參數與路徑
    num_epochs = train_config["num_epochs"]
    learning_rate = train_config["learning_rate"]
    batch_size = train_config["batch_size"]
    weight_decay = train_config["weight_decay"]
    device_str = train_config.get("device", "cuda:0")

    data_path = train_config["data_path"]
    model_folder = train_config["model_folder"]

    # 設定設備 (Device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"使用訓練設備: {device}")

    train_loader = get_dataloader(
        data_path=data_path, batch_size=batch_size, shuffle=True
    )
    autoencoder = AutoEncoder().to(device)

    # 3. 初始化 Optimizer
    # 根據 SOP (main.ipynb)，此處使用 NAdam
    optimizer = optim.NAdam(
        autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 4. 初始化 Trainer
    trainer = Trainer(
        model=autoencoder,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        model_folder=model_folder,
    )

    # 5. 開始訓練
    print(f"開始訓練，總 Epoch 數: {num_epochs}")
    r_f_loss_graph = trainer.train(num_epochs=num_epochs)

    # 6. (可選) 繪製與儲存 Loss 曲線
    plt.figure(figsize=(6, 4))
    plt.plot(r_f_loss_graph, label="Loss R_F", color="brown")
    plt.ylim(0, max(r_f_loss_graph) + 5 if r_f_loss_graph else 15)
    plt.title("Loss R_F Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    if r_f_loss_graph:
        plt.text(
            len(r_f_loss_graph) - 1,
            r_f_loss_graph[-1],
            f"{r_f_loss_graph[-1]:.4f}",
            fontsize=10,
            color="brown",
            verticalalignment="bottom",
            horizontalalignment="right",
        )

    plt.tight_layout()
    loss_plot_path = os.path.join(model_folder, "loss_curve.png")
    os.makedirs(model_folder, exist_ok=True)
    plt.savefig(loss_plot_path)
    print(f"訓練損失曲線圖已儲存至: {loss_plot_path}")


if __name__ == "__main__":
    main()

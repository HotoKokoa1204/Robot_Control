import datetime
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import yaml
from agilab.tku.visual_navigation_system.dataset import AlignedSequenceDataset
from agilab.tku.visual_navigation_system.lstm_model import LocationLSTM
from agilab.tku.visual_navigation_system.trainer import LSTMTrainer
from torch.utils.data import DataLoader, random_split

# 解決 OpenMP 多重載入的問題
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use("Agg")  # 解決 QThreadStorage 警告 (不需要互動式 GUI，只存檔)
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

    lstm_config = config["lstm"]

    # 取得超參數與路徑
    num_epochs = int(lstm_config["num_epochs"])
    learning_rate = float(lstm_config["learning_rate"])
    batch_size = int(lstm_config["batch_size"])
    sequence_length = int(lstm_config["sequence_length"])
    val_split = lstm_config.get("val_split", 0.2)
    device_str = lstm_config.get("device", "cuda:0")

    data_dir = lstm_config["train_tensor_data_dir"]
    info_dir = lstm_config["train_tensor_info_dir"]
    best_info_path = lstm_config["best_info_path"]
    best_features_path = lstm_config["best_features_path"]
    model_folder = lstm_config["lstm_weights_folder"]

    # 設定設備 (Device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"使用訓練設備: {device}")

    # 1. 建立 DataLoader
    full_dataset = AlignedSequenceDataset(
        data_dir=data_dir,
        info_dir=info_dir,
        ref_info_path=best_info_path,
        best_video_tensor=best_features_path,
        seq_length=sequence_length,
    )

    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    print(
        f"資料集總筆數: {dataset_size}, 訓練集筆數: {train_size}\
            , 驗證集筆數: {val_size}"
    )

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. 初始化模型
    lstm_model = LocationLSTM(
        input_size=512,
        hidden_size=lstm_config.get("hidden_size", 256),
        num_layers=lstm_config.get("num_layers", 2),
        output_size=512,
    ).to(device)

    # 3. 初始化 Optimizer (使用 Adam)
    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

    # 4. 初始化 Trainer
    trainer = LSTMTrainer(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        model_folder=model_folder,
    )

    # 5. 開始訓練
    print(f"開始訓練 LSTM，總 Epoch 數: {num_epochs}")
    train_loss, val_loss = trainer.train(num_epochs=num_epochs)

    # 6. 繪製與儲存 Loss 曲線
    max_loss = max(max(train_loss), max(val_loss))

    plt.figure(figsize=(6, 4))  # 稍微加大畫布，讓圖更清晰
    plt.plot(train_loss, label="Training Loss", color="blue")
    plt.plot(val_loss, label="Validation Loss", color="red")
    plt.title("Training & Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.ylim(0, max_loss * 1.1)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 標示訓練損失
    final_train_loss = train_loss[-1]
    plt.text(
        len(train_loss) - 1,
        final_train_loss,
        f"{final_train_loss:.6f}",
        fontsize=10,
        color="blue",
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    # 標示驗證損失
    final_val_loss = val_loss[-1]
    plt.text(
        len(val_loss) - 1,
        final_val_loss,
        f"{final_val_loss:.6f}",
        fontsize=10,
        color="red",
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    plt.tight_layout()
    tonow = datetime.datetime.now()
    folder = os.path.join(model_folder, f"{tonow.month}_{tonow.day}")
    os.makedirs(folder, exist_ok=True)
    loss_plot_path = os.path.join(folder, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"訓練損失曲線圖已儲存至: {loss_plot_path}")


if __name__ == "__main__":
    main()

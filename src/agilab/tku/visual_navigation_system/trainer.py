import datetime
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def loss_f(input_img: torch.Tensor, output_img: torch.Tensor) -> torch.Tensor:
    """
    計算重建影像與原始影像間的 Mean Squared Error (MSE) 損失。

    Args:
        input_img (torch.Tensor): 輸入原始影像。
        output_img (torch.Tensor): 模型重建之影像。

    Returns:
        torch.Tensor: MSE 損失。
    """
    return f.mse_loss(input_img, output_img)


class Trainer:
    """
    封裝 AutoEncoder 模型的訓練流程。
    包含混合精度訓練 (Mixed Precision Training)\
        與定期的模型檢查點 (Checkpoint) 儲存機制。
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        model_folder: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.model_folder = model_folder
        self.scaler = GradScaler()
        self.r_f_loss_graph: List[float] = []

    def _save_checkpoint(self, epoch: int) -> None:
        """
        將當前的模型權重儲存到指定的資料夾中。檔名格式為 `月_日_Epoch數.pth`。

        Args:
            epoch (int): 當前的 Epoch 數。
        """
        tonow = datetime.datetime.now()
        folder = os.path.join(self.model_folder, f"{tonow.month}_{tonow.day}")
        os.makedirs(folder, exist_ok=True)
        model_file = os.path.join(folder, f"{tonow.month}_{tonow.day}_{epoch}.pth")
        torch.save(self.model.state_dict(), model_file)
        print(f"模型已儲存至 {model_file}")

    def train(self, num_epochs: int) -> List[float]:
        """
        執行完整的訓練流程。

        Args:
            num_epochs (int): 訓練的總 Epoch 數量。

        Returns:
            List[float]: 紀錄每個 Epoch 的總 Loss_F，可用於後續繪製訓練曲線。
        """
        try:
            from tqdm import tqdm
        except ImportError:
            print("可安裝 tqdm 套件以顯示進度條")

            def tqdm(iterable, **kwargs):
                return iterable

        # 決定 autocast 的 device type
        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        total_batches = len(self.train_loader) * num_epochs

        # 使用 tqdm 包裝，設定總長度為 total_batches
        pbar = tqdm(total=total_batches, desc="模型訓練總進度", unit="batch")

        for epoch in range(num_epochs):
            loss_f_total = 0.0  # 用於累計該 Epoch 損失值
            self.model.train()

            for batch in self.train_loader:
                batch_data = batch[0].to(self.device)

                # 若使用半精度浮點數進行訓練，則可以轉換格式
                if device_type == "cuda":
                    batch_data = batch_data.to(torch.float16)

                self.optimizer.zero_grad()

                # 使用自動混合精度進行前向傳播與損失計算
                with autocast(device_type=device_type):
                    f_vector, reconstructed_f = self.model(batch_data)
                    loss = loss_f(batch_data, reconstructed_f)

                # 反向傳播並更新參數
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 累加該 Batch 的損失
                loss_f_total += loss.item()

                # 每處理完一個 batch 就更新一次進度條，這樣秒數就會即時跳動！
                pbar.update(1)
                # 更新進度條右側顯示的當前 Epoch 與 Loss 數值
                pbar.set_postfix(
                    {
                        "Epoch": f"{epoch + 1}/{num_epochs}",
                        "Loss": f"{loss_f_total:.4f}",
                    }
                )

            # 紀錄該 Epoch 的總損失
            self.r_f_loss_graph.append(loss_f_total)

            # 讓每一回合的 Loss 固定印在終端機歷史中，不會被進度條刷掉
            tqdm.write(f"Epoch {epoch + 1}：Loss_F={loss_f_total:.4f}")

            # 每 30 個 Epoch 儲存一次模型權重
            if (epoch + 1) % 30 == 0:
                self._save_checkpoint(epoch + 1)

        pbar.close()
        print("\n訓練完成！")
        return self.r_f_loss_graph


class LSTMTrainer:
    """
    封裝 LocationLSTM 模型的訓練流程。
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        model_folder: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.model_folder = model_folder
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.criterion = nn.MSELoss()

    def _save_checkpoint(self, epoch: int) -> None:
        import datetime

        tonow = datetime.datetime.now()
        folder = os.path.join(self.model_folder, f"{tonow.month}_{tonow.day}")
        os.makedirs(folder, exist_ok=True)
        model_file = os.path.join(folder, f"{tonow.month}_{tonow.day}_{epoch}.pth")
        torch.save(self.model.state_dict(), model_file)
        print(f"模型已儲存至 {model_file}")

    def train(self, num_epochs: int) -> Tuple[List[float], List[float]]:
        try:
            from tqdm import tqdm
        except ImportError:
            print("可安裝 tqdm 套件以顯示進度條")

            def tqdm(iterable, **kwargs):
                return iterable

        total_batches = len(self.train_loader) * num_epochs
        pbar = tqdm(total=total_batches, desc="LSTM 模型訓練總進度", unit="batch")

        for epoch in range(num_epochs):
            self.model.train()
            sum_train_loss, n_train = 0.0, 0

            for batch in self.train_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)

                self.optimizer.zero_grad()

                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                self.optimizer.step()

                bs = x.size(0)
                sum_train_loss += loss.item() * bs
                n_train += bs

                pbar.update(1)
                pbar.set_postfix({"Epoch": f"{epoch + 1}/{num_epochs}"})

            epoch_train_loss = sum_train_loss / n_train

            self.model.eval()
            sum_val_loss, n_val = 0.0, 0

            with torch.no_grad():
                for batch in self.val_loader:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    bs = x.size(0)
                    sum_val_loss += loss.item() * bs
                    n_val += bs

            epoch_val_loss = sum_val_loss / n_val if n_val > 0 else 0.0

            self.train_loss_history.append(epoch_train_loss)
            self.val_loss_history.append(epoch_val_loss)
            tqdm.write(
                f"Epoch {epoch + 1}/{num_epochs}：Train Loss\
                    = {epoch_train_loss:.6f}, Val Loss = {epoch_val_loss:.6f}"
            )

            if (epoch + 1) % 30 == 0:
                self._save_checkpoint(epoch + 1)

        pbar.close()

        print("\nLSTM 訓練完成！")
        return self.train_loss_history, self.val_loss_history

import torch
import torch.nn as nn


class LocationLSTM(nn.Module):
    """
    根據時間序列特徵預測下一個位置特徵的 LSTM 模型。
    """

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        output_size: int = 512,
    ):
        super(LocationLSTM, self).__init__()

        # 定義 LSTM 層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0.0,
        )

        # 定義全連接層，將隱藏狀態映射到特徵維度
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。

        Args:
            x (torch.Tensor): 輸入的時間序列張量\
                ，形狀為 (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: 預測的下一個特徵向量，形狀為 (batch_size, output_size)
        """
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)

        # 取最後一個時間步的輸出
        last_time_step_out = lstm_out[:, -1, :]

        # 進行全連接層預測
        prediction = self.fc(last_time_step_out)

        return prediction

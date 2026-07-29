import torch.nn as nn
import torch.nn.functional as f


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # 卷積與池化層
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(3, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout2d(p=0.7)

        # 拉平後的 Dense 到 512 維向量
        self.fc = nn.Linear(128 * 9 * 16, 512)

    def forward(self, x):
        # 第一個卷積和池化
        x = self.bn1(self.pool1(f.relu(self.conv1(x))))
        # 第二個卷積和池化
        x = self.bn2(self.pool2(f.relu(self.conv2(x))))
        # 第三個卷積
        x = self.bn3(self.pool3((f.relu(self.conv3(x)))))
        # 增加 Dropout防止過擬合
        x = self.dropout(x)
        # 拉平張量
        x = x.reshape(x.size(0), -1)
        # 全連接層
        f_vector = self.fc(x)

        return f_vector


class FDecoder(nn.Module):
    def __init__(self):
        super(FDecoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(512, 128 * 9 * 16), nn.ReLU())
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=3, padding=0, output_padding=0
            ),  # 27, 48
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 54 x 96
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 108 x 192
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
        )

    def forward(self, out3):
        x = self.fc1(out3)
        x = x.view(-1, 128, 9, 16)  # 調整為 128 x 9 x 16
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final_conv(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.f_decoder = FDecoder()

    def forward(self, x):
        # Encoder 提取特徵
        f_vector = self.encoder(x)
        # FDecoder 重建 f_vector
        reconstructed_f = self.f_decoder(f_vector)

        return f_vector, reconstructed_f

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

batch_size = 64
nz = 100
noise_std = 0.7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
dataset = MNIST(
    root="datasets/", train=True, download=True, transform=transforms.ToTensor()
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

sample_x, _ = next(iter(dataloader))
n_classes = len(torch.unique(dataset.targets))
w, h = sample_x.shape[-2:]
image_size = w * h
print("batch shape:", sample_x.shape)
print("width:", w)
print("height:", h)
print("image size:", image_size)
print("num classes:", n_classes)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_size + n_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self._eye = torch.eye(n_classes, device=device)  # 条件ベクトル生成用の単位行列

    def forward(self, x, labels):
        labels = self._eye[labels]  # 条件(ラベル)をone-hotベクトルに
        x = x.view(batch_size, -1)  # 画像を1次元に
        x = torch.cat([x, labels], dim=1)  # 画像と条件を結合
        y = self.net(x)
        return y


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._linear(nz, 128),
            self._linear(128, 256),
            self._linear(256, 512),
            nn.Linear(512, image_size),
            nn.Sigmoid(),  # 濃淡を0~1に
        )

    def _linear(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1, nz)
        y = self.net(x)
        y = y.view(-1, 1, w, h)  # 784 -> 1x28x28
        return y


eye = torch.eye(n_classes, device=device)


def make_noise(labels):
    labels = eye[labels]
    labels = labels.repeat_interleave(nz // n_classes, dim=-1)
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    z = z + labels
    return z


# 間違ったラベルの生成
def make_false_labels(labels):
    diff = torch.randint(1, n_classes, size=labels.size(), device=device)
    fake_labels = (labels + diff) % n_classes
    return fake_labels


fake_labels = torch.zeros(batch_size, 1).to(device)
real_labels = torch.ones(batch_size, 1).to(device)
criterion = nn.BCELoss()


def train(netD, netG, optimD, optimG, n_epochs, write_interval=1):
    # 学習モード
    netD.train()
    netG.train()

    for epoch in range(1, n_epochs + 1):
        for X, labels in dataloader:
            X = X.to(device)  # 本物の画像
            labels = labels.to(device)  # 正しいラベル
            false_labels = make_false_labels(labels)  # 間違ったラベル

            # 勾配をリセット
            optimD.zero_grad()
            optimG.zero_grad()

            # Discriminatorの学習
            z = make_noise(labels)  # ノイズを生成
            fake = netG(z)  # 偽物を生成
            pred_fake = netD(fake, labels)  # 偽物を判定
            pred_real_true = netD(X, labels)  # 本物&正しいラベルを判定
            pred_real_false = netD(X, false_labels)  # 本物&間違ったラベルを判定
            # 誤差を計算
            loss_fake = criterion(pred_fake, fake_labels)
            loss_real_true = criterion(pred_real_true, real_labels)
            loss_real_false = criterion(pred_real_false, fake_labels)
            lossD = loss_fake + loss_real_true + loss_real_false
            lossD.backward()  # 逆伝播
            optimD.step()  # パラメータ更新

            # Generatorの学習
            fake = netG(z)  # 偽物を生成
            pred = netD(fake, labels)  # 偽物を判定
            lossG = criterion(pred, real_labels)  # 誤差を計算
            lossG.backward()  # 逆伝播
            optimG.step()  # パラメータ更新

        print(f"{epoch:>3}epoch | lossD: {lossD:.4f}, lossG: {lossG:.4f}")


netD = Discriminator().to(device)
netG = Generator().to(device)
optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)
n_epochs = 30

print("初期状態")
train(netD, netG, optimD, optimG, n_epochs)

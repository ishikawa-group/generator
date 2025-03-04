# %%
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
    """Conditional Discriminator network for CGAN.

    This network determines whether an input image is real or fake, considering
    the conditional label information. It consists of convolutional layers for
    feature extraction and fully connected layers for classification.

    Attributes:
        conv: Convolutional layers for feature extraction
        fc: Fully connected layers for classification
        _eye: Identity matrix for one-hot encoding of labels
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            self._conv_layer(1, 16, 4, 2, 1),
            self._conv_layer(16, 32, 4, 2, 1),
            self._conv_layer(32, 64, 3, 2, 0),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + n_classes, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        self._eye = torch.eye(n_classes, device=device)  # 条件ベクトル生成用の単位行列

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """Create a convolutional layer with batch normalization and ReLU activation.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolving kernel
            stride (int): Stride of the convolution
            padding (int): Padding added to all sides of the input

        Returns:
            nn.Sequential: A sequential container of Conv2d, BatchNorm2d, and ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        """Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input images
            labels (torch.Tensor): Class labels for the conditional input

        Returns:
            torch.Tensor: Probability that the input is real (1) or fake (0)
        """
        x = self.conv(x)  # 特徴抽出
        labels = self._eye[labels]  # 条件(ラベル)をone-hotベクトルに
        x = torch.cat([x, labels], dim=1)  # 画像と条件を結合
        y = self.fc(x)
        return y


class Generator(nn.Module):
    """Conditional Generator network for CGAN.

    This network generates fake images based on random noise and conditional
    information. It uses transposed convolution layers to upsample the input
    noise into an image.

    Attributes:
        net: Sequential container of transposed convolution layers
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._convT(nz, 128, 3, 1, 0),
            self._convT(128, 64, 3, 2, 0),
            self._convT(64, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def _convT(self, in_channels, out_channels, kernel_size, stride, padding):
        """Create a transposed convolutional layer with batch normalization and ReLU activation.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolving kernel
            stride (int): Stride of the convolution
            padding (int): Padding added to all sides of the input

        Returns:
            nn.Sequential: A sequential container of ConvTranspose2d, BatchNorm2d, and ReLU
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """Forward pass of the generator.

        Args:
            x (torch.Tensor): Input noise tensor

        Returns:
            torch.Tensor: Generated image
        """
        x = x.view(-1, nz, 1, 1)
        y = self.net(x)
        return y


eye = torch.eye(n_classes, device=device)


def make_noise(labels):
    """Generate noise vectors with conditional information.

    Args:
        labels (torch.Tensor): Class labels to condition the noise on

    Returns:
        torch.Tensor: Noise vectors with embedded conditional information
    """
    labels = eye[labels]
    labels = labels.repeat_interleave(nz // n_classes, dim=-1)
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    z = z + labels
    return z


# 画像描画
def write(netG, n_rows=1, size=64):
    """Generate and display a grid of images using the generator.

    Args:
        netG (Generator): The generator network
        n_rows (int, optional): Number of rows in the output grid. Defaults to 1.
        size (int, optional): Size of each output image. Defaults to 64.

    Returns:
        numpy.ndarray: Grid of generated images in numpy array format
    """
    n_images = n_rows * n_classes
    z = make_noise(torch.tensor(list(range(n_classes)) * n_rows))
    images = netG(z)
    images = transforms.Resize(size)(images)
    img = torchvision.utils.make_grid(images, n_images // n_rows)
    img = img.permute(1, 2, 0).cpu().numpy()  # 画像を表示可能な形式に変換


# 間違ったラベルの生成
def make_false_labels(labels):
    """Generate incorrect labels for training the discriminator.

    Args:
        labels (torch.Tensor): Original class labels

    Returns:
        torch.Tensor: Modified labels that are different from the input labels
    """
    diff = torch.randint(1, n_classes, size=labels.size(), device=device)
    fake_labels = (labels + diff) % n_classes
    return fake_labels


fake_labels = torch.zeros(batch_size, 1).to(device)
real_labels = torch.ones(batch_size, 1).to(device)
criterion = nn.BCELoss()


def train(netD, netG, optimD, optimG, n_epochs, write_interval=1):
    """Train the CGAN model.

    Args:
        netD (Discriminator): The discriminator network
        netG (Generator): The generator network
        optimD (torch.optim.Optimizer): Optimizer for the discriminator
        optimG (torch.optim.Optimizer): Optimizer for the generator
        n_epochs (int): Number of training epochs
        write_interval (int, optional): Interval for generating sample images. Defaults to 1.
    """
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
        if write_interval and epoch % write_interval == 0:
            write(netG)


netD = Discriminator().to(device)
netG = Generator().to(device)
optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)
n_epochs = 30

print("初期状態")
write(netG)
train(netD, netG, optimD, optimG, n_epochs)

# %%
import matplotlib.pyplot as plt


def write_from_label(netG, label, n_images=10, size=64):
    """Generate and display images from a specific label using the generator.

    Args:
        netG (Generator): The generator network
        label (list): Label vector to condition the generation
        n_images (int, optional): Number of images to generate. Defaults to 10.
        size (int, optional): Size of each output image. Defaults to 64.
    """
    labels = torch.tensor([label] * n_images).to(device)
    labels = labels.repeat_interleave(nz // n_classes, dim=-1)
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    z = z + labels
    images = netG(z)
    images = transforms.Resize(size)(images)
    img = torchvision.utils.make_grid(images, len(z))
    img = img.permute(1, 2, 0).cpu().numpy()  # 画像を表示可能な形式に変換
    plt.imshow(img)
    plt.axis("off")
    plt.show()


label = [0, 0, 0, 0, 0, 10, 0, 0, 0, 0]
write_from_label(netG, label)
# %%
label = [0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0]
write_from_label(netG, label)
# %%
label = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
write_from_label(netG, label)
# %%
label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
write_from_label(netG, label)

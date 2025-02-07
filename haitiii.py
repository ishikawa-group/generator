import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ase.build import fcc111

# -----------------------------
# Step 1: データセットを作成 (atomic_numbers のみを予測対象にする)
# -----------------------------
num_samples = 10
atomic_numbers_list = []

for _ in range(num_samples):
    surf = fcc111('Pt', size=(2,2,3), vacuum=10.0)
    atomic_numbers_list.append(surf.get_atomic_numbers())

def make_samples(atomic_numbers_list):
    samples = []
    # 今回はエネルギーを扱わないので rank だけ割り当て
    # (例として ascending で0から振る)
    for i, nums in enumerate(atomic_numbers_list):
        samples.append({
            'atomic_numbers': nums,
            'rank': i  # 0,1,2,...
        })
    return samples

samples = make_samples(atomic_numbers_list)

# -----------------------------
# Step 2: データセットクラス
# -----------------------------
class AtomicDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # atomic_numbers を特徴ベクトルに
        atomic_numbers = torch.tensor(self.samples[idx]['atomic_numbers'], dtype=torch.float32)
        rank = torch.tensor(self.samples[idx]['rank'], dtype=torch.long)
        return atomic_numbers, rank

dataset = AtomicDataset(samples)
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# -----------------------------
# Step 3: CGANを定義し、原子番号を生成
# -----------------------------
nz = 100        # ノイズ次元
noise_std = 0.7 # ノイズの標準偏差
sample_x, sample_rank = next(iter(dataloader))

n_classes = len(torch.unique(sample_rank))
image_size = len(sample_x[0])  # 原子番号ベクトルの次元数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One-hot用
eye = torch.eye(n_classes, device=device)

# -----------------------------
# Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_size + n_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # x: (batch_size, image_size)
        # labels: (batch_size)
        labels_onehot = eye[labels]
        x = torch.cat([x.view(x.size(0), -1), labels_onehot], dim=1)
        return self.net(x)

# -----------------------------
# Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nz + n_classes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, image_size),
            nn.ReLU()  # 原子番号なので ReLUに(整数化は別で行っても可)
        )

    def forward(self, z, labels):
        labels_onehot = eye[labels]
        x = torch.cat([z, labels_onehot], dim=1)
        return self.net(x)

netD = Discriminator().to(device)
netG = Generator().to(device)

optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# -----------------------------
# ノイズ生成
# -----------------------------
def make_noise(labels):
    # labels: (batch_size)
    labels_onehot = eye[labels].repeat_interleave(nz // n_classes, dim=1)
    # ノイズ生成
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    # ラベル情報を加味
    return z + labels_onehot

real_labels = torch.ones(batch_size, 1, device=device)
fake_labels = torch.zeros(batch_size, 1, device=device)
n_epochs = 30

# -----------------------------
# トレーニングループ
# -----------------------------
def train(netD, netG, optimD, optimG, n_epochs):
    netD.train()
    netG.train()
    for epoch in range(1, n_epochs + 1):
        for atomic_numbers, ranks in dataloader:
            atomic_numbers = atomic_numbers.to(device)
            ranks = ranks.to(device)

            # ----- Discriminator 学習 -----
            z = make_noise(ranks)
            fake_data = netG(z, ranks)
            pred_fake = netD(fake_data, ranks)
            pred_real = netD(atomic_numbers, ranks)

            lossD_fake = criterion(pred_fake, fake_labels)
            lossD_real = criterion(pred_real, real_labels)
            lossD = lossD_fake + lossD_real

            optimD.zero_grad()
            lossD.backward()
            optimD.step()

            # ----- Generator 学習 -----
            z = make_noise(ranks)
            fake_data = netG(z, ranks)
            pred = netD(fake_data, ranks)
            lossG = criterion(pred, real_labels)

            optimG.zero_grad()
            lossG.backward()
            optimG.step()

        print(f"Epoch [{epoch}/{n_epochs}] | LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")

train(netD, netG, optimD, optimG, n_epochs)

# -----------------------------
# サンプル生成
# -----------------------------
def generate_samples(netG, rank, n_samples=5):
    labels = torch.tensor([rank]*n_samples, dtype=torch.long, device=device)
    z = make_noise(labels)
    fake_samples = netG(z, labels)
    return fake_samples

rank_to_generate = 0
generated = generate_samples(netG, rank_to_generate, 5)
print(f"Generated atomic numbers (rank={rank_to_generate}): {generated}")
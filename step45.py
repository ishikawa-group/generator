import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 提供されたデータ
samples = [
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78]), 'adsorption_energy': -0.1849839673041687, 'rank': 5},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78]), 'adsorption_energy': -0.12388265139058607, 'rank': 6},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78]), 'adsorption_energy': -0.1849839673041687, 'rank': 5},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78]), 'adsorption_energy': -0.09300284921151018, 'rank': 7},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78]), 'adsorption_energy': -0.07815589528275577, 'rank': 8},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78]), 'adsorption_energy': 0.3188310614203995, 'rank': 9},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78]), 'adsorption_energy': 0.7602581926263335, 'rank': 10}
]

# カスタムデータセット
class AtomicDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        atomic_numbers = torch.tensor(self.samples[idx]['atomic_numbers'], dtype=torch.float32)
        adsorption_energy = torch.tensor(self.samples[idx]['adsorption_energy'], dtype=torch.float32)
        rank = torch.tensor(self.samples[idx]['rank'], dtype=torch.long)  # ランクは整数値として扱う
        return atomic_numbers, adsorption_energy, rank

# データローダー
batch_size = 2  # 任意のバッチサイズに変更できます
dataset = AtomicDataset(samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 最初のサンプルを取得
sample_x, sample_energy, sample_rank = next(iter(dataloader))

# ここでの処理
n_classes = len(torch.unique(torch.tensor([sample['rank'] for sample in samples])))  # ユニークなrankの数を取得
image_size = len(sample_x[0])  # atomic_numbers の長さを画像のサイズとみなす（例えば、12）

# サンプルの表示
print(f"Sample atomic numbers: {sample_x}")
print(f"Sample adsorption energy: {sample_energy}")
print(f"Sample rank: {sample_rank}")
print(f"Number of classes (unique ranks): {n_classes}")
print(f"Image size (atomic numbers length): {image_size}")

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DiscriminatorとGeneratorの定義
class Discriminator(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1 + n_classes, 512),  # +1は吸着エネルギーの次元
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._eye = torch.eye(n_classes, device=device)

    def forward(self, x, energy, rank):
        rank = self._eye[rank.long()]  # rankをlong型に変換
        energy = energy.unsqueeze(1)  # energyを2次元に変換
        x = torch.cat([x, energy, rank], dim=1)
        y = self.net(x)
        return y

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1 + n_classes, 128),  # +1は吸着エネルギーの次元
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
        self._eye = torch.eye(n_classes, device=device)

    def forward(self, x, energy, rank):
        rank = self._eye[rank.long()]  # rankをlong型に変換
        energy = energy.unsqueeze(1)  # energyを2次元に変換
        x = torch.cat([x, energy, rank], dim=1)
        return self.net(x)

# ハイパーパラメータ
input_dim = 12  # atomic_numbersの次元
z_dim = 100
batch_size = 2
lr = 0.0002
n_epochs = 30

# モデルの初期化
netD = Discriminator(input_dim, n_classes).to(device)
netG = Generator(z_dim, input_dim, n_classes).to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr)
optimizerG = optim.Adam(netG.parameters(), lr=lr)

# トレーニングループ
for epoch in range(n_epochs):
    for atomic_numbers, adsorption_energy, rank in dataloader:
        atomic_numbers = atomic_numbers.to(device)
        adsorption_energy = adsorption_energy.to(device)
        rank = rank.to(device)

        # 本物のラベルと偽物のラベル
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Discriminatorの学習
        z = torch.randn(batch_size, z_dim).to(device)
        fake_energy = torch.randn(batch_size, 1).to(device)
        fake_rank = torch.randint(0, n_classes, (batch_size,), dtype=torch.long).to(device)
        fake_rank = netG._eye[fake_rank]  # rankをone-hotエンコーディング
        fake_data = netG(z, fake_energy, fake_rank)

        optimizerD.zero_grad()
        real_output = netD(atomic_numbers, adsorption_energy, rank)
        fake_output = netD(fake_data.detach(), fake_energy, fake_rank)
        d_loss_real = criterion(real_output, real_labels)
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizerD.step()

        # Generatorの学習
        optimizerG.zero_grad()
        fake_output = netD(fake_data, fake_energy, fake_rank)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizerG.step()

    print(f'Epoch [{epoch+1}/{n_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

# 画像生成関数
def generate_samples(netG, energy, rank, n_samples=10):
    z = torch.randn(n_samples, z_dim).to(device)
    energy = torch.tensor([energy] * n_samples).to(device)
    rank = torch.tensor([rank] * n_samples, dtype=torch.long).to(device)
    rank = netG._eye[rank]  # rankをone-hotエンコーディング
    samples = netG(z, energy, rank)
    return samples

# 特定の吸着エネルギーとrankに基づいてサンプルを生成
energy_to_generate = -1.0
rank_to_generate = 1
generated_samples = generate_samples(netG, energy_to_generate, rank_to_generate)
print(generated_samples)
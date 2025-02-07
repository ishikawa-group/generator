import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 提供されたデータ
samples = [
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 79]), 'adsorption_energy': -0.1849839673041687, 'rank': 0},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 79, 78]), 'adsorption_energy': -0.12388265139058607, 'rank': 1},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 78, 79, 78, 78]), 'adsorption_energy': -0.1849839673041687, 'rank': 2},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 78, 79, 78, 78, 78]), 'adsorption_energy': -0.09300284921151018, 'rank': 3},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 78, 79, 78, 78, 78, 78]), 'adsorption_energy': -0.07815589528275577, 'rank': 4},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 78, 79, 78, 78, 78, 78, 78]), 'adsorption_energy': 0.3188310614203995, 'rank': 5},
    {'atomic_numbers': np.array([78, 78, 78, 78, 78, 79, 78, 78, 78, 78, 78, 78]), 'adsorption_energy': 0.7602581926263335, 'rank': 6}
]
nz = 100
noise_std = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self._eye = torch.eye(n_classes, device=device) # 条件ベクトル生成用の単位行列

    def forward(self, x, labels):
        labels = self._eye[labels] # 条件(ラベル)をone-hotベクトルに
        x = x.view(batch_size, -1) # 画像を1次元に
        x = torch.cat([x, labels], dim=1) # 画像と条件を結合
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
            nn.Sigmoid() # 濃淡を0~1に
        )

    def _linear(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1, nz)
        y = self.net(x)
        y = y.view(-1, 1, w, h) # 784 -> 1x28x28
        return y
    
eye = torch.eye(n_classes, device=device)
def make_noise(labels):
    labels = eye[labels]
    labels = labels.repeat_interleave(nz // n_classes, dim=-1)
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    z = z + labels
    return z


# 画像描画
def write(netG, n_rows=1, size=64):
    n_images = n_rows * n_classes
    z = make_noise(torch.tensor(list(range(n_classes)) * n_rows))
    images = netG(z)
    images = transforms.Resize(size)(images)
    img = torchvision.utils.make_grid(images, n_images // n_rows)
    img = img.permute(1, 2, 0).cpu().numpy()  # 画像を表示可能な形式に変換
    plt.imshow(img)
    plt.axis('off')
    plt.show()

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

    for epoch in range(1, n_epochs+1):
        for X, labels in dataloader:
            X = X.to(device) # 本物の画像
            labels = labels.to(device) # 正しいラベル
            false_labels = make_false_labels(labels) # 間違ったラベル

            # 勾配をリセット
            optimD.zero_grad()
            optimG.zero_grad()

            # Discriminatorの学習
            z = make_noise(labels) # ノイズを生成
            fake = netG(z) # 偽物を生成
            pred_fake = netD(fake, labels) # 偽物を判定
            pred_real_true = netD(X, labels) # 本物&正しいラベルを判定
            pred_real_false = netD(X, false_labels) # 本物&間違ったラベルを判定
            # 誤差を計算
            loss_fake = criterion(pred_fake, fake_labels)
            loss_real_true = criterion(pred_real_true, real_labels)
            loss_real_false = criterion(pred_real_false, fake_labels)
            lossD = loss_fake + loss_real_true + loss_real_false
            lossD.backward() # 逆伝播
            optimD.step() # パラメータ更新

            # Generatorの学習
            fake = netG(z) # 偽物を生成
            pred = netD(fake, labels) # 偽物を判定
            lossG = criterion(pred, real_labels) # 誤差を計算
            lossG.backward() # 逆伝播
            optimG.step() # パラメータ更新

        print(f'{epoch:>3}epoch | lossD: {lossD:.4f}, lossG: {lossG:.4f}')
        if write_interval and epoch % write_interval == 0:
            write(netG)

netD = Discriminator().to(device)
netG = Generator().to(device)
optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)
n_epochs = 30

print('初期状態')
write(netG)
train(netD, netG, optimD, optimG, n_epochs)
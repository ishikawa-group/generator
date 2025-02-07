# A script for predict the adsorption energy with GAN.
# Step 1. Prepare the dataset by generating the surface structures and calculating the adsorption energy.
# Step 2. Make the descriptor-target pair, by setting atomic numbers as descriptors and adsorption energy as target.
# Step 3. Assign the strongest adsorption energy surfaces as "rank=1".
# Step 4. Put above dataset and train the CGAN.
# Step 5. Generate fake-samples for "rank=1" surfaces, to generate the surfaces with stronger adsorption energy.

num_samples = 10
e_ads = []
atomic_numbers = []

# --- Step 1. Prepare the dataset by generating the surface structures and calculating the adsorption energy.
from ase.build import fcc111
import numpy as np

def get_adsorption_energy(surface):
    # 仮
    return np.random.normal(0, 1)

for i in range(num_samples):
    surf = fcc111('Pt', size=(2, 2, 3), vacuum=10.0)
    e_ad = get_adsorption_energy(surface=surf)  # function to be implemented
    atomic_numbers.append(surf.get_atomic_numbers())  # get element information for surface

    print(f"Adsorption energy of sample {i+1}: {np.random.normal(0, 1)}")
    e_ads.append(e_ad)

# --- Step 2. Make the descriptor-target pair, by setting atomic numbers as descriptors and adsorption energy as target.
# samples = make_samples(atomic_numbers, e_ads)
def make_samples(atomic_numbers, e_ads):
    samples = []
    for atomic_number, e_ad in zip(atomic_numbers, e_ads):
        sample = {
            'atomic_numbers': atomic_number,
            'adsorption_energy': e_ad
        }
        samples.append(sample)
    return samples

samples = make_samples(atomic_numbers,e_ads)


# --- Step 3. Assign the strongest adsorption energy surfaces as "rank=1".
# samples = assign_rank(samples)
def assign_rank(samples):
    # sort samples by adsorption energy
    samples = sorted(samples, key=lambda x: x['adsorption_energy'])
    # assign rank
    for i, sample in enumerate(samples):
        sample['rank'] = i + 1
    return samples

samples = assign_rank(samples)

print(samples)



import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
# --- Step 4. Put above dataset and train the CGAN.
# train_cgan()
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
nz = 100
noise_std = 0.7
# サンプルを表示して確認
sample_x, sample_energy, sample_rank = next(iter(dataloader))
print(f"Sample atomic numbers: {sample_x}")
print(f"Sample adsorption energy: {sample_energy}")
print(f"Sample rank: {sample_rank}")

n_classes = len(torch.unique(torch.tensor([sample['rank'] for sample in samples])))  # ユニークなrankの数を取得
image_size = len(sample_x[0])  # atomic_numbers の長さを画像のサイズとみなす（例えば、12）

# サンプルの表示
print(f"Sample atomic numbers: {sample_x}")
print(f"Sample adsorption energy: {sample_energy}")
print(f"Sample rank: {sample_rank}")
print(f"Number of classes (unique ranks): {n_classes}")
print(f"Image size (atomic numbers length): {image_size}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        

netD = Discriminator().to(device)
netG = Generator().to(device)
optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)
n_epochs = 30

print('初期状態')
train(netD, netG, optimD, optimG, n_epochs)